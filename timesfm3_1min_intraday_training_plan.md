> **TIMESFM-3 / 1MIN INTRADAY PILOT**

# TimesFM-3 1min 日内 Return

## 训练与验证计划（v1）

目标：先在现有 1min return + features 数据上建立一个可复现、可评估的 TimesFM-3 基线，优先验证多变量协同、动态 context 与 64min horizon 是否能形成稳定的增量预测价值。

| **项目**     | **v1 决策**                                                        |
|--------------|--------------------------------------------------------------------|
| 频率         | 1min，单日常规交易时段 390min                                      |
| 主 Target    | 现有 1min return（先冻结定义与时间戳语义）                         |
| Context      | C_min = 64，C_max = 192；实际长度动态变化                          |
| Horizon      | 64min / 64 points                                                  |
| Past-only    | 约 15-20 条市场状态 / microstructure / vol / cross-market features |
| Past-future  | 约 3-5 条未来已知时间结构：TOD、time-to-close、已知事件等          |
| 单次变量预算 | 推荐 Target + Covariates <= 32；首版目标约 20-25                  |
| Session      | 首版不跨 overnight；只在当日有完整未来 64min 时生成样本            |

> **核心原则**  
> C_max 是“最多使用多少真实历史”，不是固定输入宽度。某时刻只有 83min 可用，就输入 83min；TimesFM 内部只为 32-point patch 对齐做少量 masked left-padding，而不是人为补到 192min。

## 1. 试验目标与成功标准

- 先验证 TimesFM-3 在 1min intraday 数据上的 zero-shot / lightweight adaptation 价值，而不是一开始就做复杂 fine-tuning。
- 确认 return-only、+past-only、+past-future 三档输入的边际增益。
- 确认动态 context（64-192min）是否优于固定短 context，并检查长历史是否真的有用。
- 最终评价以 out-of-sample IC / rank IC、方向、quantile calibration 和带成本 PnL/utility 为主，而不是只看 MSE。

## 2. 数据与变量定义

每个样本对应某个 intraday 决策时刻 t。所有 target 与 covariates 必须被严格对齐到同一 1min grid，并且在实际决策时刻可获得。首版不跨 overnight，以避免把收盘到次日开盘误当成连续 1min 时间。

### 2.1 Target：先用 1min return

> **主 Target**  
> y_t = 1min return。优先直接使用现场已经稳定生产的 return 序列；在开始训练前冻结 price source、simple/log return、单位（如 bps）、分钟边界、延迟与缺失值处理。

若 target 为 1min return，模型输出的是未来 64 个逐分钟 return：ŷ\_{t+1}, …, ŷ\_{t+64}。需要 5/10/30/60min 累计方向时，可以先对逐分钟预测求和；第二阶段再和 direct multi-horizon target 做对照。

### 2.2 Past-only covariates：未来未知的市场状态

这类变量是主要的辅助预测信息：当前和历史可见，但未来真实值在 t 时刻未知。它们不要求和 target 同量纲，也不要求数量与 target 相等。首版建议从已有 features 中筛 15-20 条，覆盖互补的经济含义。

| **Feature family** | **候选变量**                                                                      | **首版建议** |
|--------------------|-----------------------------------------------------------------------------------|--------------|
| Return / momentum  | 5m / 15m / 30m trailing return；短期 momentum / reversal；相对 VWAP / open 的位置 | 2-4          |
| Volatility         | 5m / 15m / 30m realized vol；abs return；intraday range                           | 2-4          |
| Liquidity / flow   | spread、book imbalance、OFI、bid/ask depth、trade imbalance、volume、trade count  | 6-10         |
| Market / sector    | 指数 1min return、sector return、cross-asset state、beta-adjusted residual state  | 2-4          |

筛选原则：宁可 15 条高质量、低冗余的状态序列，也不要 50 条高度相关的变体。每一条 variate 都会进入跨变量 attention，因此“独立信息价值”比“feature 数量”更重要。

### 2.3 Past-future covariates：未来已知的时间/日历结构

这类变量在 t 时刻已经知道未来 64min 的值，因此可以把 context+future 全部提供给模型。金融场景中通常数量很少，重点是日内时钟与预先确定的事件。

| **变量**         | **说明**                                      | **优先级** |
|------------------|-----------------------------------------------|------------|
| sin(time_of_day) | 日内周期的连续编码                            | 推荐       |
| cos(time_of_day) | 与 sin 配对，避免 390min 边界不连续           | 推荐       |
| time_to_close    | 归一化到 \[0,1\] 或稳定尺度                   | 推荐       |
| known_event_flag | 已知发布时间/auction/rebalance 等             | 按需要     |
| session phase    | 若用 one-hot，每个 flag 都占一个 variate slot | 可选       |

> **变量预算**  
> 首版建议：1 target + 16~20 past-only + 3 past-future = 20~24 variates。尽量保持单次完整联合建模在 32 variates 以内，避免 chunking 破坏 cross-variate interaction。

## 3. 动态 Context 与样本生成

### 3.1 C_min / C_max 规则

```text
C_min = 64
C_max = 192
H = 64

C_t = min(C_max, available_intraday_history(t))
only create sample if C_t >= C_min
require future 64min to remain inside the same trading session
```

这意味着模型在日内逐步“长大 context”：刚满足 C_min 时用 64min，随后可能用 83、127、168min，达到 192min 后只保留最近 192min。模型内部再将实际长度向上对齐到 32 的倍数。

| **真实历史** | **输入点数** | **Temporal patches** | **Padding**                       | **定位**             |
|--------------|--------------|----------------------|-----------------------------------|----------------------|
| 64min        | 64           | 2                    | 不需要额外真实历史；仅 patch 对齐 | 最短允许 context     |
| 83min        | 83           | 3（内部 pad 到 96）  | 13 个 masked 左 padding           | 正常使用             |
| 128min       | 128          | 4                    | 无                                | 短 context 基线      |
| 192min       | 192          | 6                    | 无                                | v1 主上限            |
| 256min       | 256          | 8                    | 无                                | 后续 ablation        |
| 284min       | 284          | 9（内部 pad 到 288） | 4 个 masked 左 padding            | 可行，但单日样本更少 |

### 3.2 为什么不固定补到 C_max

1. **保留上午样本**：如果强制每个样本都必须有 192/256min 真实历史，会大量丢掉上午可用窗口。动态 context 只要求达到 C_min。

2. **避免“假历史”**：masked padding 可以安全对齐 patch，但它不会创造信息。60min 真实历史补成 256min 并不会等价于 256min context。

3. **贴近线上使用**：线上每分钟滚动一次，实际可用历史本来就是随时间增长；动态 context 与部署路径一致。

### 3.3 样本数量直觉

若只要求 C_min=64 且 H=64，同一 390min session 中理论可用 anchor 数约为 390 - 64 - 64 + 1 ≈ 263（具体取决于分钟时间戳定义）。相比要求完整 192/256min context，动态 context 能保留显著更多上午样本。

> **注意独立样本数**  
> 相邻分钟的滑窗高度重叠，所以“窗口数量”不等于独立样本数量。更重要的是交易日数量、市场 regime 覆盖和严格的 chronological split。

## 4. 模型内部如何使用这三类变量

在 TimesFM-3 中，target、past-only 与 past-future 会沿 variate 轴拼到同一个 2D token grid 中。它们共享 patch encoder、Temporal Attention、Variate Attention 和 output head，核心区别来自未来信息是否被 mask。

```text
1min aligned series
├─ target [U, C]
├─ past-only covariates [V, C]
└─ past-future covars [W, C+H]
↓ concatenate variates
[U+V+W, time]
↓ patchify (32 points = 32min)
token grid [variates × temporal patches × 1280]
↓ 20 × Mixing Transformer
Temporal causal attention → Variate full attention → FFN
↓ output head 1280 → 64×9 quantiles
↓ return only target rows
```

- Target：未来 horizon 全部未知，被 CPM mask；模型预测其未来 continuation。
- Past-only：未来同样未知，因此未来部分也被 mask；只作为历史条件信息。
- Past-future：未来值已知，因此未来 64min 可以直接进入 token construction，帮助模型理解日内位置与已知 schedule。
- 在 1min 频率下，32-point input patch = 32min；64-point output patch = 64min。C=64~192 对应 2~6 个 temporal tokens。

## 5. 训练 / 适配执行计划

首版优先做“可解释的逐步增量实验”，避免一开始同时改变 target、context、feature set 与训练方式。这样每一步都能回答一个清晰问题。

| **阶段**              | **动作**                                                                                         | **产出**                            |
|-----------------------|--------------------------------------------------------------------------------------------------|-------------------------------------|
| A. 数据冻结           | 冻结 1min target 定义、timestamp、missing policy、session 边界；建立统一 1min grid               | 数据审计报告 + schema               |
| B. Zero-shot baseline | TimesFM-3 仅输入 1min return；动态 context 64-192；H=64                                          | return-only 基线                    |
| C. + Past-only        | 加入 15-20 条精选市场状态 features                                                               | 验证多变量增益                      |
| D. + Past-future      | 加入 TOD sin/cos、time-to-close 等未来已知变量                                                   | 验证日内时钟增益                    |
| E. Calibration        | 在 TimesFM 输出上训练 Ridge / LightGBM / small MLP 校准层；输入 median/quantile width + 少量状态 | 提升交易可用性                      |
| F. Direct targets     | 增加 R_5m/R_15m/R_30m/R_60m target 作为对照                                                      | 比较“1min 累计” vs “direct horizon” |
| G. Base adaptation    | 只有前面证明确有价值后，再评估 source-level fine-tuning / adaptation                             | 可选，单独立项                      |

### 5.1 第一轮实验矩阵

| **ID** | **Input**                | **C_min** | **C_max** | **H** | **问题**               |
|--------|--------------------------|-----------|-----------|-------|------------------------|
| E0     | Return only              | 64        | 192       | 64    | 建立 TimesFM 基线      |
| E1     | Return + 16~20 past-only | 64        | 192       | 64    | 验证市场状态增益       |
| E2     | E1 + TOD/time-to-close   | 64        | 192       | 64    | 验证 future-known 增益 |
| E3     | E2                       | 64        | 128       | 64    | 短 context ablation    |
| E4     | E2                       | 64        | 256       | 64    | 长 context ablation    |
| E5     | E2                       | 96        | 192       | 64    | 提高最小历史门槛       |

### 5.2 Feature selection 原则

- 所有筛选只在 training period 内完成；不要用 test IC / importance 选 feature。
- 优先按 family 选代表变量，再看相关性与增量 IC，减少 highly correlated variants。
- 维持总 variates 约 20-25；如超过 32，优先删冗余 covariates，而不是依赖随机 chunking。
- 任何 rolling normalization / z-score / realized vol 计算都只能使用当时之前的数据。

## 6. 训练集切分与评测

### 6.1 数据切分

- 按交易日期 chronological split；禁止 random window split。
- 首版以整日为 split 单位，避免 train/valid/test 在同一天共享高度重叠的窗口。
- 若未来需要在日内切 split，边界至少 purge 最大 horizon（64min），并考虑 context overlap。
- 不开启 overnight continuation；开盘 gap 作为新的 session。

### 6.2 评测维度

| **维度**         | **切片**                            | **指标**                                                  |
|------------------|-------------------------------------|-----------------------------------------------------------|
| Forecast quality | 1/5/10/20/30/60min                  | MAE / RMSE（辅助）；逐 horizon IC / rank IC               |
| Direction        | 5/10/30/60min                       | sign accuracy；long/short conditional mean                |
| Distribution     | 全部 horizon                        | q10-q90 coverage；quantile crossing；interval width       |
| Trading utility  | 策略持有期                          | 成本后 PnL、turnover、hit ratio、drawdown、capacity proxy |
| Context buckets  | 64-95 / 96-127 / 128-191 / 192+     | 确认 context 越长是否真的更好                             |
| Regime           | open / midday / close；high/low vol | 检查模型是否只在少数时段有效                              |

> **Baseline 必须保留**  
> 至少同时跑 Ridge / LightGBM（同一特征集、同一 split），并保留简单的 last-return / zero-return baseline。Foundation model 的价值必须以相同数据和交易约束下的增量表现来证明。

## 7. 上线前的关键检查项

| **检查项**          | **验收标准**                                                                 | **状态** |
|---------------------|------------------------------------------------------------------------------|----------|
| Target 对齐         | 1min return 的起止分钟、price source、可用时间明确；预测 t+1 时不含 t+1 信息 | □        |
| Feature causal      | 所有 rolling / cross-sectional feature 只使用可获得数据                      | □        |
| Future-known 真实性 | past-future 只包含在 t 时刻确实知道未来值的变量                              | □        |
| Session             | 样本不跨 overnight；最后 64min 不生成需要越过收盘的 label                    | □        |
| Dynamic context     | 只补到下一个 32-point boundary，不固定补到 C_max                             | □        |
| Variable budget     | 单次完整联合建模尽量 <=32 variates                                          | □        |
| Split               | 按日期 chronological；test period 未参与 feature selection / tuning          | □        |
| Cost model          | 交易评测包含 spread、fees、slippage、latency                                 | □        |
| Reproducibility     | 固定数据版本、feature schema、模型 checkpoint、随机种子                      | □        |

## 8. v1 参数建议（可直接开工）

```text
frequency = 1min
target = 1min_return
C_min = 64
C_max = 192
horizon = 64
input_patch = 32      # TimesFM-3 checkpoint default
output_patch = 64     # TimesFM-3 checkpoint default
past_only = 16~20 selected features
past_future = [sin_tod, cos_tod, time_to_close] (+ optional known-event flags)
total_variates ≈ 20~24 (keep <=32 when possible)
session_rule = intraday only; no overnight crossing
split = chronological by trading date
```

> **Go / No-Go 决策**  
> 如果 E1/E2 在多个独立 test periods 中无法稳定超过 Ridge/LightGBM，先不要投入 base-model fine-tuning；优先回到 target 定义、feature 信息含量和时间对齐。若在 30-60min horizon 上出现稳定增益，再推进 direct multi-horizon target、校准层和更长 context。

**附：模型约束与实现依据**

> **Official implementation references**  
> google-research/timesfm: src/timesfm3/torch/model.py（32/64 patch、CPM、output head、stitching）；src/timesfm3/torch/transformer.py（Temporal + Variate Attention）；src/timesfm3/torch/timesfm3_forecaster.py（dynamic context 与 API shapes）；src/timesfm3/torch/evaluator.py（32 variates/forward benchmark chunking）。正式商业/生产使用前需单独确认 TimesFM-3.0 权重许可。
