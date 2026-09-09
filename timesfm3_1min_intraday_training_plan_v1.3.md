> **TIMESFM-3 / 1MIN INTRADAY PILOT**

# TimesFM-3 1min 日内 Return

## 训练与验证计划（v1.3 · research-updated）

目标：先在现有 1min return + features 数据上建立一个可复现、可评估的 TimesFM-3 基线，在不破坏 pretrained forecasting topology 的前提下，验证多变量协同、动态 context、dense-anchor supervision 与 64min horizon 是否能形成稳定的增量预测价值。

> 本次 v1.3 更新重点：
>
> 1. 把“公开 inference semantics”和“推断的 pretraining task construction”严格分开；
> 2. 加入 `F0-final / F0-all / F1 / F1-MV` 四级训练路线；
> 3. 将 **all-eligible-anchor Pinball + final-anchor cumulative Huber** 提升为当前主候选；
> 4. 明确 `decode()` 不能直接训练，以及公开 Torch `forward()` 是 inference-specialized 的工程 caveat；
> 5. 加入 shifted next-64 labels、CPM、segment packing、role supervision set Ω 等我们最新的训练机制理解。

---

## 0. 核心业务设定

| 项目 | v1.3 决策 |
|---|---|
| 频率 | 1min，单日常规交易时段约 390min |
| 主 Target | 现有 1min return；先冻结定义与 timestamp semantics |
| Context | `C_min=64`, `C_max=192`；线上实际长度动态变化 |
| Horizon | 64min / 64 points |
| Input patch | 32 points = 32min |
| Output patch | 64 points = 64min |
| Quantiles | q0.1 ... q0.9，共 9 个 |
| Past-only | 约 15-20 条市场状态 / microstructure / vol / cross-market features |
| Past-future | 约 3-5 条未来已知时间结构：TOD、time-to-close、已知事件等 |
| 单次变量预算 | 推荐 Target + Covariates <= 32；首版目标约 20-25 |
| Session | 首版不跨 overnight；所有训练 label 不跨 session |
| 主 fine-tune 候选 | `F1 = all-eligible-anchor Pinball + 0.3 final cumulative Huber` |

> **核心原则**  
> `C_max` 是“最多使用多少真实历史”，不是固定输入宽度。某时刻只有 83min 可用，就输入 83min；TimesFM 内部只为 32-point patch 对齐做 masked left-padding，而不是人为补到 192min。

---

## 1. 试验目标与成功标准

第一阶段不是追求复杂训练，而是回答几个严格可证伪的问题：

- TimesFM-3 zero-shot 在 1min return 上是否有基本预测价值；
- `return-only -> +past-only -> +past-future` 是否存在稳定边际增益；
- 动态 context 64-192min 是否优于固定短 context；
- dense-anchor forecasting 是否比 final-anchor-only 更有效利用有限训练数据；
- cumulative business loss 是否能提升 5/15/30/60min IC / RankIC / cost-adjusted PnL；
- auxiliary past-only forecasting 是否真的 regularize 主 target，而不是让容易预测的 covariates 主导 loss。

最终成功标准以 out-of-sample：

```text
IC / RankIC
sign accuracy
quantile calibration
prediction decile monotonicity
cost-adjusted PnL / turnover / drawdown
```

为主，而不是只看 MSE 或 total validation loss。

---

## 2. 数据与变量定义

每个业务 sample 对应某个 intraday 决策时刻 `t`。所有 target 与 covariates 必须严格对齐到同一 1min grid，并且满足真实可用时间约束。

首版不跨 overnight，避免把收盘到次日开盘误当成连续 1min 时间。

### 2.1 Target：1min return

主 target：

\[
y_t = \text{1min return}
\]

优先直接使用现场已经稳定生产的 return 序列；训练前冻结：

```text
price source
simple/log return
单位（如 bps）
分钟边界
决策 latency
missing / halt policy
```

模型 final anchor 的 64min 输出：

```text
y_hat[t+1], ..., y_hat[t+64]
```

每个 future minute 输出 9 个 quantiles。

业务累计 horizon：

\[
R_h = \sum_{i=1}^{h} y_{t+i},\qquad h\in\{5,15,30,60\}
\]

首版继续以 1min return path 为主 target，再通过累计构造业务 horizon；direct multi-horizon targets 放到后续对照实验。

### 2.2 Past-only covariates：未来未知的市场状态

定义：在 anchor `t` 时可看到历史，但 `t+1...t+H` 的真实值未知。

候选：

| Feature family | 候选变量 | 首版建议 |
|---|---|---:|
| Return / momentum | 5m / 15m / 30m trailing return；momentum / reversal；相对 VWAP / open | 2-4 |
| Volatility | 5m / 15m / 30m realized vol；abs return；intraday range | 2-4 |
| Liquidity / flow | spread、book imbalance、OFI、bid/ask depth、trade imbalance、volume、trade count | 6-10 |
| Market / sector | 指数 1min return、sector return、cross-asset state、beta-adjusted residual | 2-4 |

筛选原则：

```text
宁可 15 条高质量、低冗余状态序列
也不要 50 条高度相关变体
```

每个 variate 都会进入跨变量 attention；独立信息价值比 feature 数量更重要。

### 2.3 Past-future covariates：未来已知结构

判断标准：

> 在 anchor `t` 时，我是否已经知道该变量在 `t+1...t+64` 的真实值？

若知道，才能放 past-future。

推荐：

| 变量 | 说明 | 优先级 |
|---|---|---|
| `sin(time_of_day)` | 日内周期连续编码 | 推荐 |
| `cos(time_of_day)` | 与 sin 配对 | 推荐 |
| `time_to_close` | 归一化稳定尺度 | 推荐 |
| `known_event_flag` | 已知发布时间 / auction / rebalance | 按需要 |
| `session_phase` | one-hot 会占多个 variate slots | 可选 |

禁止误放：

```text
future realized OFI
future spread
future realized vol
future volume
```

它们虽然训练数据里“事后知道”，但在真实 anchor 当时并不知道，属于 leakage。

### 2.4 变量预算

首版：

```text
1 target
+ 16~20 past-only
+ 3~4 past-future
= ~20~25 variates
```

尽量保持单次完整联合建模 `<=32 variates`，优先删冗余 feature，而不是依赖 variate chunking。

---

## 3. Dynamic Context 与样本生成

### 3.1 C_min / C_max

```text
C_min = 64
C_max = 192
H = 64

C_t = min(C_max, available_intraday_history(t))
only create business sample if C_t >= C_min
require future 64min to remain inside the same trading session
```

模型 context 随日内时间逐步增长：

```text
64 -> 83 -> 127 -> 168 -> 192 -> 192 -> ...
```

达到 192 后保留最近 192min。

### 3.2 patch alignment

| 真实历史 | 输入点数 | Temporal patches | 内部 left padding | 定位 |
|---|---:|---:|---:|---|
| 64min | 64 | 2 | 0 | 最短允许 context |
| 83min | 83 | 3 | 13 | 正常使用 |
| 128min | 128 | 4 | 0 | 短 context 基线 |
| 192min | 192 | 6 | 0 | v1 主上限 |
| 256min | 256 | 8 | 0 | 后续 ablation |
| 284min | 284 | 9 | 4 | 可行，但有效日内窗口更少 |

### 3.3 为什么不固定补到 C_max

- 保留上午样本；
- 避免制造“假历史”；
- 贴近线上真实可用 context；
- TimesFM 原生 mask 已支持 patch 对齐 padding。

### 3.4 样本数量直觉

若只要求 `C_min=64`, `H=64`，390min session 理论业务 anchors 约：

\[
390-64-64+1\approx263
\]

具体取决于时间戳定义。

但相邻窗口高度重叠，因此窗口数不等于独立样本数；真正重要的是：

```text
交易日数量
regime 覆盖
严格 chronological split
purge / embargo
```

---

## 4. TimesFM-3 内部如何使用三类变量

高层语义：

```text
1min aligned series
├─ target [U, C]
├─ past-only [V, C]
└─ past-future [W, C+H]
      ↓ concatenate variates
[U+V+W, time]
      ↓ patchify 32
2D token grid [V_total × N × 1280]
      ↓ 20 × MixingTransformer
Temporal causal attention
-> Variate full attention
-> FFN
      ↓
output head 1280 -> 64 × 9
      ↓
raw logits [B, V_total, N, 64, 9]
```

高层 `TimesFM3Forecaster` 最终只切出 target rows 返回。

### 4.1 target / past-only / past-future 的核心区别

- **Target**：future unknown；horizon masked；用户明确要求返回 forecast。
- **Past-only**：future unknown；horizon 同样 masked；主要用于 conditioning。
- **Past-future**：future known；未来值可进入 input construction。

从 backbone 的角度，target 与 past-only 都属于：

```text
future-unavailable variates
```

而不是“一个是真正时序，一个只是 feature”。

### 4.2 raw output 的 variate 对齐

输入 slot 顺序固定：

```text
0 ... U-1             targets
U ... U+V-1           past-only
U+V ... U+V+W-1       past-future
```

网络各层保持 variate slot，因此 raw output 也是相同顺序。

结论：

```text
input variate j -> raw output variate j
```

不需要额外 target-output matching。

---

## 5. 对 TimesFM-3 训练机制的最新理解

这一节是本次 v1.3 更新的核心。

### 5.1 证据等级

后续所有判断分成：

```text
Confirmed       公开代码直接支持
Strong inference 多处实现痕迹一致，但训练 loop 未公开
Unknown         不能从公开材料确认
```

### 5.2 公开 PyTorch 是 inference-only，但明显保留 training lineage

公开 `TimesFM3Torch` 文件明确标记：

```text
inference only
forward(): equivalent to Flax __call__
```

早期测试中甚至有：

```python
test_forward_pass_training_dict
```

其 inputs 包含：

```text
values
masks
patch_segment_ids
patch_positions
patch_is_target
patch_is_past_only
patch_is_past_future_covariate
```

这更像 pretraining batch schema，而不是最终用户 inference API。

因此当前最合理的理解：

```text
internal Flax/JAX training-capable implementation
        ↓ port / parity
public PyTorch inference implementation
```

### 5.3 shifted next-64 labels：dense forecasting 的关键线索

`get_output_patch_via_roll()` 的公开 docstring 明确说：

```text
Creates labels of output_patch length by rolling the patched inputs.
```

对 `P_in=32`, `P_out=64`, `rolls=2`：

```text
P0 -> P1 + P2
P1 -> P2 + P3
P2 -> P3 + P4
...
```

即：

\[
h_j \rightarrow \text{next 64 points}
\]

这强烈支持：

\[
\boxed{\text{多个 temporal tokens 都可以作为 forecast origin}}
\]

而不是训练时只监督最后一个 token。

### 5.4 training topology 与 inference topology 一致

inference stitch path：

```text
last context token       -> future 1..64
first future masked token -> future 33..96
second future masked token -> future 65..128
```

因此：

```text
training: token j -> next 64 points
inference: last real token -> next 64 points
```

是同一个 output topology。

### 5.5 CPM 的正确定位

CPM 更应理解为：

```text
改变哪些 patches / variates 可见
```

而不是改变模型结构。

TimesFM-3 inference 创建：

```text
history | MASK | MASK | ...
```

然后一次 full-sequence forward。

它不是：

```text
预测 y1 -> 填回 y1 -> 再预测 y2
```

而是 **causal latent-state propagation**。

因此：

```text
non-autoregressive output-value generation
+ causal hidden-state mixing
```

同时成立。

### 5.6 random CPM pretraining：目前属于强推断，不是官方确认

公开 TimesFM-3 有 `patch_cpm_mask` 和 CPM-compatible inference，但没有完整 training loop。

TiRex 公开论文提供了一个非常相似的机制参照：

```text
random contiguous patch masking
shifted future labels
causal flow
loss on multiple output tokens
```

但这只能作为机制类比，不能写成“TimesFM-3 官方一定照 TiRex 训练”。

### 5.7 segment_ids 暗示 sequence packing

Transformer 支持：

```text
segment_ids
segment_pos
```

running RevIN 也支持 segment boundary reset。

这非常适合 foundation pretraining packing：

```text
| series A | series B | series C |
```

且不同 segment 不互相 attention。

判断：

- 模型支持 segmented full-sequence training：**Confirmed**；
- 大规模 pretraining 很可能使用 packing：**Strong inference**；
- 实际 packing policy：**Unknown**。

### 5.8 真正未知的是 supervision set Ω

网络会为所有 variates 输出 raw logits，但公开资料没有告诉我们真实 pretraining 中：

\[
\Omega=\text{哪些 variate × token × horizon × quantile 坐标进入 loss}
\]

当前置信度：

| 项目 | 判断 |
|---|---|
| 9 quantiles 监督 | 很高 |
| shifted next-64 labels | 很高 |
| 多 token forecasting supervision | 高 |
| target rows 有 loss | 极高 |
| past-only 有 auxiliary loss | 中等~中高 |
| past-future 主要作为 conditional input | 高 |
| role assignment 动态采样 | 中等 |
| role-specific loss weight | 中等 |

因此不能把 `past-only auxiliary loss` 写成官方事实。

---

## 6. 为什么我们的 fine-tuning 训练方式要单独设计

我们的目标不是：

```text
猜测并复刻一个未公开的 foundation pretraining recipe
```

而是：

```text
保留 pretrained forecasting topology
+
利用 dense shifted supervision
+
把最终业务梯度集中到 return alpha
```

因此采用分级路线：

```text
F0-final
-> F0-all
-> F1
-> F1-MV
```

---

## 7. 四级训练目标

### 7.1 F0-final：final-anchor Pinball

这是最干净、最 deployment-consistent 的 baseline。

设 final anchor 为 `T`：

\[
L_{F0-final}
=
\frac1{64\times9}
\sum_{h=1}^{64}\sum_q
\rho_q(y_{T+h}-\hat y_{T,h,q})
\]

必须先跑它，因为它验证：

```text
数据
mask
output slicing
loss
backprop
部署 graph
```

是否全部正确。

### 7.2 F0-all：all-eligible-anchor Pinball

对一个完整 training window 内所有 eligible temporal anchors 做监督：

\[
L_{F0-all}
=
\frac1{|\mathcal J|}
\sum_{j\in\mathcal J}
\frac1{64\times9}
\sum_{h,q}
\rho_q(y_{j+h}-\hat y_{j,h,q})
\]

推荐的 eligibility：

```text
real_history_at_anchor >= C_min
next_64 label 全部在同一 session
future label valid
不含 halt / invalid
future-known covariates 在该历史 anchor 当时也确实已知
```

#### C=192 的例子

```text
context: P0 P1 P2 P3 P4 P5
future : P6 P7
```

shifted labels：

```text
P0 -> P1 P2
P1 -> P2 P3
P2 -> P3 P4
P3 -> P4 P5
P4 -> P5 P6
P5 -> P6 P7
```

但若严格要求每个 anchor 至少有 64min visible history，则推荐：

```text
eligible = P1, P2, P3, P4, P5
```

而 P0 只有 32min visible history，可作为 “pretraining-like all-valid-token” 单独 ablation，不和 strict business training 混淆。

### 7.3 F1：主推荐版本

\[
L_{F1}
=
L_{F0-all}
+
0.3L_{cum,final}
\]

对 final anchor q50 path：

\[
\hat R_h=\sum_{i=1}^h\hat y_{T+i,0.5}
\]

\[
R_h=\sum_{i=1}^h y_{T+i}
\]

\[
h\in\{5,15,30,60\}
\]

每个 horizon 使用 training-only robust scale `s_h`：

\[
L_{cum,final}
=
\frac14\sum_h
Huber\left(\frac{\hat R_h-R_h}{s_h}\right)
\]

首版：

```text
lambda_cum = 0.3
```

validation ablation：

```text
0.1 / 0.3 / 0.5
```

#### 为什么 cumulative loss 只放 final anchor

```text
all-anchor Pinball -> dense generic forecasting geometry
final cumulative Huber -> real trading decision alignment
```

如果 cumulative loss 也在全部历史 anchors 上重复，会：

- 对高度重叠 windows 反复计权；
- 放大 business objective；
- 更容易破坏 pretrained distribution forecasting geometry。

因此首版明确：

\[
\boxed{\text{forecast supervision dense，business supervision sparse}}
\]

### 7.4 F1-MV：selected past-only auxiliary

仅在 F1 稳定后：

\[
L_{F1-MV}
=
L_{F1}
+
0.05L_{aux}
\]

首批 auxiliary 只选少数：

```text
realized volatility
spread / liquidity state
volume state
```

高噪声 OFI / imbalance 不默认加入。

Past-future 不做 forecast loss，因为其 future 已知且已经作为 conditioning input。

---

## 8. Dense-anchor supervision 如何缓解数据量不足

final-anchor：

```text
1 window -> 1 forecast origin
```

dense-anchor：

```text
1 window -> K eligible forecast origins
```

当 `C=192`, `C_min=64`，通常 K≈5。

这不会创造新的独立市场样本，但能提升：

```text
label efficiency
每次 forward 的监督密度
不同 intraday anchors 的梯度利用率
```

因此更准确的说法是：

\[
\boxed{\text{增加监督密度，而不是增加独立样本数}}
\]

这正是长 context 不一定导致训练“有效数据骤减”的重要缓解因素。

---

## 9. 工程实现：不要直接训练 `decode()`

当前公开 `decode()` 带：

```python
@torch.no_grad()
```

所以不能直接：

```python
loss(model.decode(...))
loss.backward()
```

业务 fine-tune 需要：

```text
直接使用 differentiable model.forward()
+
自己构造 values / masks / patch_is_target / patch_cpm_mask
+
自己构造 shifted labels 与 anchor loss mask
```

### 9.1 另一个关键 caveat：公开 `forward()` 是 inference-specialized

当前 Torch `forward()` 对 fully-masked patches 使用 leading-mask-only `cumprod` 逻辑，代码注释明确说明这是 inference 行为；同时又暗示内部 Flax `__call__` 在 training 下有不同分支。

因此：

> **不要把当前公开 Torch forward 的 mask semantics 直接等同于完整官方 pretraining semantics。**

我们分两条路线：

#### A. deployment-consistent fine-tune

```text
真实 context + suffix masked horizon
```

优先实现 F0-final。

#### B. pretraining-like full-sequence fine-tune

显式实现：

```text
causal full sequence
shifted next-64 labels
eligible anchor mask
必要时 random CPM
training / inference mask semantics 区分
```

F0-all / F1 属于这一条。

开发顺序：

```text
F0-final 正确
-> F0-all
-> F1
-> F1-MV
-> random CPM
```

---

## 10. All-token training 的 leakage audit

对任意历史 anchor `t_j`，都必须用当时的真实信息集构造输入。

### 10.1 Past-only

`future values unknown`，不得让 anchor 看到未来真实值。

### 10.2 Past-future

只有 deterministic / scheduled future 才可见。

### 10.3 左 padding

padding 只能：

```text
mask input
mask invalid anchor
```

不能进入 loss。

### 10.4 session boundary

任何 shifted next-64 label 若跨：

```text
overnight
halt discontinuity
session reset
```

必须 loss weight = 0。

---

## 11. 训练执行阶段

| 阶段 | 动作 | 目的 |
|---|---|---|
| A | 数据 / timestamp / latency / session semantics 冻结 | 防止后续实验不可解释 |
| B | Zero-shot return-only | 建立 foundation baseline |
| C | + 16~20 past-only | 验证多变量状态增益 |
| D | + TOD / time-to-close | 验证 future-known 增益 |
| E | classical calibration：Ridge / LightGBM / small MLP | 判断 backbone 表达是否可用 |
| F | F0-final | 验证可微 fine-tune graph |
| G | F0-all | 验证 dense shifted supervision |
| H | F1 | **主业务微调候选** |
| I | F1-MV | selected auxiliary ablation |
| J | random CPM / role sampling | 仅后续研究，不是 v1 必做 |

---

## 12. 第一轮完整实验矩阵

### 12.1 输入 / context ablation

| ID | Input | C_min | C_max | H | 问题 |
|---|---|---:|---:|---:|---|
| E0 | Return only | 64 | 192 | 64 | TimesFM zero-shot 基线 |
| E1 | Return + 16~20 past-only | 64 | 192 | 64 | 市场状态增益 |
| E2 | E1 + TOD/time-to-close | 64 | 192 | 64 | future-known 增益 |
| E3 | E2 | 64 | 128 | 64 | 短 context ablation |
| E4 | E2 | 64 | 256 | 64 | 长 context ablation |
| E5 | E2 | 96 | 192 | 64 | C_min sensitivity |

### 12.2 fine-tuning loss ablation

| ID | Training graph | Loss | 定位 |
|---|---|---|---|
| **T0 / F0-final** | suffix-mask deployment graph | final-anchor return Pinball | 必跑 control |
| **T1 / F0-all** | causal full sequence | all-eligible-anchor return Pinball | dense supervision 主实验 |
| **T2 / F1** | causal full sequence | F0-all + `0.3 cumulative Huber(final)` | **当前主候选** |
| **T3 / F1-MV** | same as T2 | F1 + `0.05 selected past-only aux` | multi-task ablation |
| **T4** | T2 + random CPM | same as F1 | pretraining-like mask ablation，后续 |

### 12.3 第一轮不默认加入

```text
quantile-crossing loss
direction loss
ranking loss
all-covariate auxiliary loss
random role reassignment
```

这些只在更基础的 distribution / cumulative 指标已经合理后再研究。

---

## 13. Feature selection 原则

- 所有 feature selection 只在 training period 内完成；
- 优先按经济 family 选代表变量；
- 冗余变量优先删除；
- rolling normalization / z-score / realized vol 只能使用历史数据；
- test period 不能参与 feature selection；
- 超过 32 variates 时先减 feature，不把 benchmark chunking 当 production 方案。

---

## 14. Loss 细节

### 14.1 Pinball

对 quantile `q`：

\[
\rho_q(e)=\max(qe,(q-1)e)
\]

主 target：

```text
q = 0.1 ... 0.9
h = 1 ... 64
```

### 14.2 cumulative Huber scale

`5/15/30/60min` 的累计 return 分布尺度不同，因此不能直接裸平均。

推荐：

```text
scale[h] = training-period MAD
```

或稳定 std。

严禁使用 validation / test 统计量。

### 14.3 为什么不首选 MSE

1min return heavy-tail / jump 多，MSE 更容易被少数极端点主导。

Pinball + Huber 更符合当前业务：

```text
distribution forecasting
+
robust cumulative alignment
```

---

## 15. Split 与 purge

- 按交易日期 chronological split；
- 禁止 random window split；
- 首版以整日为 split 单位；
- 若日内切 split，至少 purge 最大 label horizon = 64min；
- 进一步考虑 context overlap；
- 不跨 overnight continuation。

Dense-anchor supervision 不能改变 split 原则：同一窗口里有多个 anchors 仍然高度相关。

---

## 16. 评测

| 维度 | 切片 | 指标 |
|---|---|---|
| Forecast quality | 1/5/10/20/30/60min | MAE / RMSE 辅助；IC / RankIC |
| Direction | 5/15/30/60min | sign accuracy；conditional mean |
| Distribution | 全 horizon | q10-q90 coverage；crossing；interval width |
| Trading utility | 策略持有期 | 成本后 PnL、turnover、hit ratio、drawdown |
| Context buckets | 64-95 / 96-127 / 128-191 / 192+ | context length sensitivity |
| Intraday | open / midday / close | 时段稳健性 |
| Regime | high / low vol | regime robustness |

必须保留：

```text
zero-return
last-return / simple momentum
Ridge
LightGBM
```

等 classical baselines。

---

## 17. Checkpoint selection

不要单按 total validation loss。

至少记录：

```text
IC / RankIC @ 5,15,30,60
sign accuracy
prediction decile monotonicity
q10-q90 coverage
quantile crossing rate
cost-adjusted PnL
turnover
max drawdown
```

如果 F1-MV total loss 更低，但 return IC / PnL 没提高，则不采用。

原则：

\[
\boxed{\text{主 target 的可交易增量优先于“所有变量都预测得更准”}}
\]

---

## 18. 上线前检查项

| 检查项 | 验收标准 | 状态 |
|---|---|---|
| Target 对齐 | 1min return 起止、price source、可用时间明确 | □ |
| Feature causal | 所有 rolling / cross-sectional features 只使用可获得数据 | □ |
| Future-known 真实性 | past-future 在 anchor 当时确实知道 future | □ |
| Session | label 不跨 overnight / session reset | □ |
| Dynamic context | 只 pad 到 32-point boundary | □ |
| Variable budget | 尽量 <=32 variates | □ |
| Split | chronological；test 不参与 selection | □ |
| Loss mask | padding / invalid / halt / session 越界 = 0 | □ |
| Anchor eligibility | all-token loss 只覆盖合法历史与未来标签 | □ |
| Auxiliary supervision | 若开启，记录变量、权重、scale | □ |
| Cost model | spread / fee / slippage / latency | □ |
| Reproducibility | 固定数据、schema、checkpoint、seed | □ |

---

## 19. 可直接开工的 v1.3 参数

```text
frequency = 1min
target = 1min_return

C_min = 64
C_max = 192
horizon = 64

input_patch = 32
output_patch = 64
quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

past_only = 16~20 selected features
past_future = [sin_tod, cos_tod, time_to_close] (+ optional known-event flags)
total_variates ~= 20~25
keep <=32 when possible

session_rule = intraday only; no overnight crossing
split = chronological by trading date

F0_final = final-anchor return Pinball
F0_all = all-eligible-anchor return Pinball
F1 = F0_all + 0.3 * final cumulative Huber @ [5,15,30,60]min
F1_MV = F1 + 0.05 * selected past-only auxiliary forecasting

anchor_min_true_history = 64min
anchor_label_horizon = 64min

checkpoint_selection = IC / RankIC / calibration / cost-adjusted PnL
not total val loss alone

raw_model_output = all variates, 9 quantiles
high_level_forecaster_output = target rows

training_api_rule = do not train through @torch.no_grad decode()
implementation = differentiable low-level forward + explicit masks + explicit labels
```

---

## 20. Go / No-Go

### 20.1 输入价值

如果 E1/E2 在多个独立 test periods 中无法稳定超过 Ridge/LightGBM：

```text
先不要投入复杂 base fine-tuning
```

优先回到：

```text
target 定义
feature 信息含量
latency / timestamp 对齐
```

### 20.2 训练路线

执行顺序：

```text
F0-final
-> F0-all
-> F1
-> F1-MV
```

只有 F0-all 稳定优于 final-only，才把 dense-anchor 设为默认；只有 F1 提升业务指标，才保留 cumulative Huber；只有 F1-MV 真正提高主 target IC / PnL，才保留 auxiliary loss。

### 20.3 当前主候选

\[
\boxed{
L
=
L_{Pinball,all\ eligible\ anchors}
+
0.3L_{CumHuber,final\ anchor}
}
\]

这是当前最值得作为 v1.3 主训练 candidate 的 objective。

---

## 21. 当前未知项与继续研究列表

仍不能从公开实现确认：

1. 官方 TimesFM-3 pretraining 是否对 past-only rows 算 loss；
2. 官方 `Ω` 包含哪些 token / variate positions；
3. random CPM mask ratio / mask length；
4. training branch 对 fully-masked patches 的 attention 语义；
5. role 是否动态采样；
6. 是否存在额外 median / point objective；
7. sequence packing 的真实使用方式。

这些应继续通过源码历史、论文、作者 release、parity experiments 验证。

---

## 22. 实现依据与参考入口

### TimesFM-3 官方仓库

- `src/timesfm3/torch/model.py`
  - inference-only 标记；
  - `forward(): equivalent to Flax __call__`；
  - 32/64 patch；
  - CPM preprocessing；
  - raw output head；
  - decode / stitching。
- `src/timesfm3/torch/util.py`
  - `get_output_patch_via_roll()` shifted labels；
  - segment-aware running stats；
  - stitching。
- `src/timesfm3/torch/transformer.py`
  - Temporal causal attention；
  - Variate attention；
  - `segment_ids / segment_pos`；
  - full-sequence / no-cache path。
- `src/timesfm3/torch/model_test.py`
  - `test_forward_pass_training_dict`；
  - training-style metadata fields。
- `src/timesfm3/torch/timesfm3_forecaster.py`
  - dynamic context；
  - high-level target-only slicing。
- `src/timesfm3/torch/evaluator.py`
  - benchmark 32-variate forward budget / chunking。

### 外部机制参考

- TiRex / Contiguous Patch Masking：用于理解 random contiguous masking + shifted causal forecasting + dense token-level supervision；仅作机制类比，不作为 TimesFM-3 官方 recipe 证据。

---

## 23. v1.3 最终训练原则

```text
1. architecture 保持不动
2. 保留 9-quantile forecasting semantics
3. 先做 final-anchor control
4. 再利用 all-eligible-anchor dense shifted supervision
5. business cumulative objective 只放 final decision anchor
6. past-only auxiliary 只做小权重 ablation
7. past-future 永远按真实未来可知性审计
8. train / deploy graph 的 mask semantics 必须显式验证
9. checkpoint 按业务指标选，不按 total loss 选
10. 不把对官方 pretraining 的推断写成事实
```

> **一句话总结**  
> 当前我们不再把 TimesFM-3 看成“给一个 context，只在最后预测一个 horizon”的普通 forecaster，而是更接近一个 **多个 temporal forecast origins 共享同一 shifted next-64 quantile head 的 causal multivariate foundation model**。业务 fine-tuning 的关键，是利用这种 dense forecasting geometry，同时把真正的交易目标稀疏地约束在 final decision anchor 上。
