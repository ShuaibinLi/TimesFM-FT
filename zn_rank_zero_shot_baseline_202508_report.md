# ZN Rank Selected100 / TimesFM-3 Zero-Shot Baseline

## 2025-08 专项评测报告

## 1. 结论摘要

本报告评测原始 TimesFM-3 checkpoint 在 ZN 规则 1min return 数据上的
zero-shot 表现。没有加载 adapter，没有更新任何模型参数。

评测结果：

- E1/E2 的 20 个 selected features 明显优于 return-only E0，但仍没有
  达到可直接交易或直接进入 5-epoch fine-tuning 的标准。
- E2 的 point-wise mean pinball 为 0.1248，较 E0 的 0.1277 改善约 2.3%。
- E2 的 pooled 60min RankIC 为正（0.0509），但 mean-daily 60min
  RankIC 为负（-0.1398）。
- E2 的 60min OOS R² vs zero-return 为 -0.4773。
- 加入 time-of-day covariates 后，E2 相对 E1 只有极小 pinball 改善，
  60min R² 反而略差。

因此当前结论是：

```text
selected features 含有增量信息
≠
zero-shot TimesFM 已能稳定利用这些信息
```

建议先增强 train-only 时间稳定性筛选，再运行 1-epoch head-only pilot；
不建议直接启动完整 5-epoch 训练。

## 2. 数据与评测契约

- 数据集：`zn_rank_selected100_1min_wmid_ticks_v1`
- 评测日期：2025-08-01 至 2025-08-29
- 完整交易日：21
- 评测 anchors：5,523
- 每日模型网格：09:31–16:00 America/New_York，共 390 分钟
- context：动态 64–192 分钟
- forecast horizon：未来 64 个 1min returns
- quantiles：9
- target price：`WMid`
- target unit：ZN ticks
- target：

```text
return_1m[t] = (wmid[t] - wmid[t-1]) / 0.015625
```

截至 anchor `t` 的历史 `return_1m` 位于模型输入第一条 variate；未来
`return_1m[t+1:t+64]` 只作为 labels。

特征选择仅使用 train split：

- train：562 日，2022-11-01 至 2025-02-07
- 原始候选：100
- missing-rate gate：不超过 5%
- correlation limit：0.9
- family limit：4
- 最终 past-only：20
- deterministic past-future：3
- 总 variates：`1 target + 20 past-only + 3 past-future = 24`

2025-08 未参与 feature selection 或任何模型拟合。

## 3. 三组 zero-shot 输入

### E0：return-only

输入只有历史 `return_1m`。用于判断 TimesFM 原始 return prior。

### E1：return + selected20

输入历史 `return_1m` 和 20 个 train-only selected past-only features。

### E2：return + selected20 + TOD

在 E1 上加入：

- `sin_time_of_day`
- `cos_time_of_day`
- `time_to_close`

这是当前推荐实验配置，但 baseline 结果显示 TOD 增益尚不稳定。

## 4. 指标定义、点数与统计可靠性

### 4.1 每天到底有多少个点

每个完整交易日有 390 个 1min bars，但不是 390 个 forecast origins：

```text
context_min = 64
horizon = 64
stride = 1

origins/day = 390 - 64 - 64 + 1 = 263
```

因此 2025-08：

- 每天每个 lead/horizon 有 263 对 prediction-target；
- 21 天共 `263 × 21 = 5,523` 个 forecast origins；
- 每个单独 lead 的 `valid_points=5,523`；
- 汇总 64 个 leads 时共有 `5,523 × 64 = 353,472` 个 point errors；
- 每个 cumulative horizon 仍是 5,523 条 path predictions，不是
  353,472 条独立样本；
- 每日 IC/RankIC 用当日 263 对样本计算。

### 4.2 点数是否足够计算 IC

263 个点可以计算日内 IC，但精度有限。若错误地假设 263 点相互独立，
零相关附近的单日 IC 标准误约为：

```text
SE ≈ 1 / sqrt(263 - 3) ≈ 0.062
单日 95% 随机波动范围约为 ±0.12
```

实际误差更大，因为：

- 相邻 anchors 的 context 高度重叠；
- h-minute cumulative target 相邻两行共享 `h-1` 个 future returns；
- 60min target 的相邻样本共享 59/60 的路径；
- feature 和 WMP return 本身存在日内自相关。

所以 pooled 的 5,523 点绝不能当作 5,523 个独立观测。60min horizon
每个交易日只有约 4.4 个不重叠的 60min 区块，这只是有效样本量的直观
下界，不是正式估计。

### 4.3 按交易日 block bootstrap

为避免把日内重叠当成独立样本，本报告对 21 个 daily RankIC 做
20,000 次按日重采样。E2 mean-daily RankIC 的 95% 区间为：

```text
horizon   mean daily RankIC   95% day-block CI   positive days
1m             -0.0124        [-0.041,  0.019]       7 / 21
5m             -0.0323        [-0.071,  0.007]       6 / 21
10m            -0.0520        [-0.110,  0.005]       9 / 21
15m            -0.0733        [-0.141, -0.009]       8 / 21
20m            -0.0675        [-0.139, -0.001]       6 / 21
30m            -0.0833        [-0.164, -0.003]       5 / 21
60m            -0.1398        [-0.222, -0.058]       6 / 21
```

解释：

- 1/5/10min 区间包含 0，当前一个月无法确认正负；
- 15/20/30/60min 区间位于 0 以下，2025-08 内表现显著偏负；
- 只有 21 天，结论只适用于这个月，不能替代完整
  2025-08—2026-01 holdout。

### 4.4 每个输出指标的含义

#### 样本与误差

- `samples`：forecast origins 数，本报告为 5,523。
- `valid_points`：所有有效 origin-lead 组合数；总体为 353,472。
- `MAE`：P50/median forecast 与真实 return 的平均绝对误差，单位为 ticks。
- `RMSE`：P50 forecast 的均方根误差，对大误差更敏感。
- `zero_return_rmse`：永远预测 0 的 RMSE。
- `last_return_rmse`：把 anchor 时最后一个已实现 return 外推到未来的 RMSE。
- `OOS R² vs zero`：`1 - model_SSE / zero_SSE`；正数表示优于预测 0，
  负数表示不如预测 0。
- `OOS R² vs last`：相对 last-return baseline 的同类指标。
- `directional_accuracy`：P50 预测与非零真实 target 符号相同的比例。

#### 分布与 quantile

- `mean_pinball`：9 个 quantiles 的平均 pinball loss；越低越好。
- `coverage_qXX`：真实值小于等于预测 qXX 的比例；理想值应接近 XX%。
- `mean_absolute_coverage_error`：9 个 empirical coverages 与 nominal
  quantiles 的平均绝对偏差；越低越好。
- `q10_q90_coverage`：真实值落在预测 Q10–Q90 区间内的比例；理想约 80%。
- `mean_q10_q90_width`：Q90-Q10 平均宽度，单位 ticks；需结合 coverage，
  不能只追求窄。
- `quantile_crossing_rate`：低 quantile 预测高于相邻高 quantile 的比例；
  理想为 0。

#### Lead 与累计 horizon

- `per_lead`：分别评估第 1、2、…、64 个未来 1min return。
- `cumulative_horizon=h`：先将前 h 个 P50 predictions 相加，并将前 h
  个真实 returns 相加，再计算指标。
- `masked_path_excluded`：因为前 h 路径中存在 missing target 而排除的
  origins 数。

#### IC 与排序

- `IC`：P50 prediction 与 target 的 Pearson correlation，关注线性关系。
- `RankIC`：二者排名的 Spearman correlation，对尺度和极端值更稳健。
- `pooled IC/RankIC`：把全部日期混合后计算，可能混入跨日 regime 差异。
- `mean_daily IC/RankIC`：每天单独计算后等权平均，更接近日内选时能力。

#### 条件收益与 deciles

- `long_conditional_mean`：P50 cumulative forecast > 0 时真实累计 return
  的均值。
- `short_conditional_mean`：forecast < 0 时真实累计 return 的均值。
- `long_short_spread`：上述二者之差；正且稳定才符合方向性业务直觉。
- `prediction_decile_means`：按预测从低到高分成十组，各组真实 return 均值。
- `prediction_decile_monotonicity`：decile 序号与 decile realized means
  的 RankIC。
- `prediction_decile_spread`：最高预测 decile 的真实均值减最低 decile。

#### Trading proxy 与 slices

- `gross_mean`：`sign(forecast) × realized cumulative return` 的均值。
- `turnover`：相邻 anchor position sign 的绝对变化；每天第一条从 0 开始。
- `net_mean`：gross 减 `cost_per_turnover × turnover`。
- `hit_rate`：proxy PnL > 0 的比例。
- `max_drawdown`：按日期/时间顺序累计 proxy PnL 的最大回撤。
- `slices`：按 context length、open/midday/close、high/low volatility
  重复计算 cumulative 指标。

当前 `cost_per_turnover=0`，且 60min positions 每分钟重叠，因此 trading
proxy 只用于研究诊断，不是可交易回测。

## 5. 总体 point/quantile 指标

```text
variant                     E0          E1          E2
mean pinball                0.1277      0.1250      0.1248
point RMSE                  0.4780      0.4765      0.4766
OOS R² vs zero             -0.0164     -0.0098     -0.0104
directional accuracy        50.87%      50.99%      50.93%
Q10–Q90 coverage            87.33%      85.42%      85.20%
mean Q10–Q90 width          1.4450      1.2997      1.2871 ticks
quantile crossing rate      0.0020%     0.0004%     0.0021%
```

解读：

- selected20 使 pinball 和 interval width 有小幅改善，说明 features
  确实改变了 zero-shot 分布预测。
- 所有版本的 point OOS R² 仍为负，预测均值不如恒等于零。
- Q10–Q90 nominal coverage 应约为 80%，实际 E2 为 85.2%，区间偏宽。
- median coverage 为 49.6%，中心位置校准尚可。
- quantile crossing 基本可以忽略。

## 6. E2 累计 horizon 指标

```text
horizon   pooled RankIC   mean-daily RankIC   OOS R² vs zero   direction
1m          0.0026            -0.0124             -0.0547        50.10%
5m          0.0080            -0.0323             -0.0796        49.81%
10m         0.0236            -0.0520             -0.1076        50.28%
15m         0.0257            -0.0733             -0.1266        50.68%
20m         0.0476            -0.0675             -0.1364        50.86%
30m         0.0613            -0.0833             -0.1924        52.34%
60m         0.0509            -0.1398             -0.4773        52.51%
```

### pooled 与 mean-daily 为什么相反

pooled RankIC 将 21 天所有 anchors 混合排序，可能利用“某一天整体预测
较高、该日整体 realized return 也较高”的跨日 regime 差异。

mean-daily RankIC 先在每个交易日内部计算 RankIC，再跨日平均，更接近
真实日内选时业务。E2 的 pooled 30/60min RankIC 为正，而 mean-daily
显著为负，说明模型主要捕获了跨日尺度差异，却不能稳定完成日内排序。

当前 checkpoint metric 已配置为 mean-daily RankIC，因此后续训练不会
被正的 pooled RankIC 误导。

## 7. E0/E1/E2 在 60min 的对比

```text
variant       pooled RankIC   mean-daily RankIC   OOS R² vs zero
E0               0.0063            -0.1953             -0.9590
E1               0.0380            -0.1458             -0.4418
E2               0.0509            -0.1398             -0.4773
```

结论：

- selected20 显著减轻 return-only 的 60min 失效，features 有价值。
- TOD 提高 pooled RankIC 和少量 mean-daily RankIC，但 R² 变差。
- E2 相对 E1 的变化很小，不能据此确认 TOD 有稳健边际价值。

## 8. 20 个 selected features

下面的 selection score 是 train split 上 5/15/30/60min 中最大的绝对
RankIC，不是 2025-08 结果。所有入选列在 train model grid 上 missing
rate 均为 0。

### 1. `zn__imp_t19_002__bidside__emsT_300s`

- family：`impulse:ActionArith`
- train score：0.02565，最强 horizon 60min
- 含义：bid-side `ActionArith` 复合脉冲的 300s time-decayed sum。
- 作用：表达慢速 bid-side action/liquidity pressure。
- 注意：`t19_002` 是内部研究编号，业务解释以 graph 算子为准。

### 2. `zn_sdi2`

- family：pivot/flip regime
- train score：0.02279，最强 horizon 60min
- 含义：sequential hyper-pivot direction state。
- 作用：表达 pivot 序列的方向性和趋势/反转 regime。

### 3. `zn__imp_t21_001__up__emsT_60s`

- family：`impulse:MessageFieldSelector`
- train score：0.02239，最强 horizon 60min
- 含义：MessageField field 4、up-side 脉冲的 60s time-decayed sum。
- 作用：保留最近一分钟特定消息字段的方向性压力。

### 4. `zn_rega_deemedlevqty_n40`

- family：book liquidity
- train score：0.02098，最强 horizon 60min
- 含义：前 40 档订单簿数量的 robust median level quantity。
- 作用：衡量深层流动性厚度，对少数异常大档位不敏感。

### 5. `zn__imp_t7_001__bidside__emsT_15s`

- family：`impulse:ActionArith`
- train score：0.02075，最强 horizon 60min
- 含义：由 action codes 50/6 构造的 bid-side 复合脉冲，15s
  time-decayed sum。
- 作用：快速 liquidity/action pressure。

### 6. `zn_sweep1_pivotdirvol_2s`

- family：volatility
- train score：0.01868，最强 horizon 60min
- 含义：2s half-life 的 pivot-direction activity/volatility。
- 作用：识别超短 pivot 活跃和微观波动 regime。

### 7. `zn_deep_kyletv_s300_d95_lam`

- family：liquidity/impact
- train score：0.01811，最强 horizon 60min
- 含义：300s window、0.95 EMA decay 的 Kyle lambda。
- 作用：估计单位成交/流量对应的价格冲击。
- 注意：低成交量时可能不稳定。

### 8. `zn_CancelSignedQueueFrac_Touch_GlobalEventN64_Imb`

- family：trade/order flow
- train score：0.01793，最强 horizon 5min
- 含义：最近 64 个全局事件内，touch cancel queue-fraction 的 signed
  imbalance。
- 作用：衡量最优档撤单方向。
- 注意：这是 event-count window，不是固定 64 秒。

### 9. `zn_pivotdeltasize`

- family：pivot/book state
- train score：0.01674，最强 horizon 60min
- 含义：active pivot 附近一档 delta 的 size change。
- 作用：反映 pivot 周边挂单增减。

### 10. `zn__imp_t3_004__signed__win_300s`

- family：`impulse:OETSelector`
- train score：0.01639，最强 horizon 15min
- 含义：OET field code 3 的 up-minus-down 300s window sum。
- 作用：表达慢速 order-entry toxicity/方向性状态。

### 11. `zn__imp_t18_002__signed__win_300s`

- family：`impulse:ActionArith`
- train score：0.01593，最强 horizon 5min
- 含义：多 action source 算术组合后的 signed 300s window sum。
- 作用：慢速复合订单行为压力。

### 12. `zn_JointSignedNotional_All_GlobalEventN64_CumMin`

- family：trade/order flow
- train score：0.01590，最强 horizon 5min
- 含义：最近 64 个全局事件中 joint signed notional path 的累计最小值。
- 作用：捕获窗口内最不利的 signed-notional excursion。

### 13. `zn_vnmom_5s`

- family：price/momentum
- train score：0.01517，最强 horizon 5min
- 含义：5s WMP drift 除以 5s WMP realized volatility。
- 作用：短期波动率归一化 momentum。
- 注意：低波动分母附近需要数值保护。

### 14. `zn_rega_deelogqtypre_n40`

- family：book liquidity
- train score：0.01469，最强 horizon 60min
- 含义：前 40 档、impact factor 0.9 的 deep log quantity pressure。
- 作用：衡量深层 bid/ask 数量失衡，log 压缩降低极端 qty 影响。

### 15. `zn__imp_t6_003__bidside__emsE_41`

- family：`impulse:ActionArith`
- train score：0.01416，最强 horizon 60min
- 含义：bid-side action composite 的 41-event exponentially decayed sum。
- 作用：活动速度自适应的订单行为状态。
- 注意：按事件衰减；安静时不会像 wall-clock 变量一样衰减。

### 16. `zn_rega_deelogordpre_n20`

- family：book liquidity
- train score：0.01359，最强 horizon 60min
- 含义：前 20 档、impact factor 0.9 的 deep log order-count pressure。
- 作用：补充 quantity pressure，区分“少量大单”和“许多小单”。

### 17. `zn__imp_t11_014__signed__emsT_300s`

- family：`impulse:TimeGatedAction`
- train score：0.01340，最强 horizon 15min
- 含义：type 1 time-gated action 的 up-minus-down 300s time-decayed sum；
  内部使用 50ms gap、最长 5s span。
- 作用：慢速聚合短 burst 的方向性 action。

### 18. `zn__imp_t11_007__up__win_60s`

- family：`impulse:TimeGatedAction`
- train score：0.01297，最强 horizon 60min
- 含义：type 2、2s gap 的 up-side time-gated action 60s window sum。
- 作用：最近一分钟单边 action burst 强度。

### 19. `zn__imp_t3_003__signed__win_60s`

- family：`impulse:OETSelector`
- train score：0.01233，最强 horizon 5min
- 含义：OET field code 2 的 up-minus-down 60s window sum。
- 作用：较快的 order-entry toxicity/方向性状态。

### 20. `zn__imp_t14_001__down__win_300s`

- family：`impulse:MessageFieldSelector`
- train score：0.01185，最强 horizon 60min
- 含义：MessageField field 8、down-side 的 300s window sum。
- 作用：特定消息字段的慢速单边累积状态。

## 9. 特征结构总结

最终 20 个 past-only features 包含：

- 7 个 impulse/action arithmetic 或 time-gated action；
- 2 个 OET signed windows；
- 2 个 message-field memories；
- 5 个 deep-book/pivot/liquidity states；
- 2 个 event-flow states；
- 1 个 normalized momentum；
- 1 个 micro-volatility state。

这批特征偏向 60–300s 慢状态，和 30/60min 累计 horizon 的业务目标一致，
但也提高了跨 regime 失稳风险。

## 10. 当前 baseline 的主要风险

### 10.1 feature selection 时间稳定性不足

当前 score 使用整个 train split 上各 horizon 的最大绝对 RankIC。它会偏好：

- 只在某个 horizon 有效的特征；
- 在早期 train regime 有效、后期反号的特征；
- 多重比较后偶然最大的特征。

2025-08 mean-daily RankIC 为负，说明下一版 selector 应在 train 内增加
按年度或 chronological folds 的 sign-stability gate。

### 10.2 WMid 与部分 feature 内部价格定义不完全相同

target 使用 `WMid`。部分 rankobjfair feature 内部使用 WMP/EnigmaWmp
或其它 book price state。虽然都因果，但 price-source 差异可能降低 IC，
必须保持 target contract 固定。

### 10.3 trading proxy 不能当作 PnL

E2 60min proxy：

- gross/net mean：0.2398
- hit rate：52.51%
- mean turnover：0.3996
- max drawdown：971.1
- transaction cost：0

该 proxy 使用 overlapping signals 且没有成本，不能解释为可交易 PnL。
mean-daily RankIC 和负 OOS R² 与其形成警告性冲突。

## 11. 后续训练建议

### Gate A：重做 train-only 稳定性筛选

- train 内按年或 chronological folds 计算 IC/RankIC；
- 要求主要 horizon 符号稳定；
- 对 horizon 多重比较做惩罚；
- 保持 missing rate、family cap 和 correlation cap；
- 冻结新的 selected20，不查看 2025-08 labels 做选择。

### Gate B：重新运行 E0/E1/E2 zero-shot

成功标准不是 pooled RankIC，而是：

- mean-daily RankIC 不显著为负；
- E1 相对 E0 在多个 horizon 有一致改善；
- E2 相对 E1 有可重复增益；
- prediction deciles 有基本单调性。

### Gate C：1-epoch head-only pilot

配置：

`configs/experiments/zn_rank_e2_pilot.json`

只在 Gate A/B 后运行。观察：

- train/validation pinball；
- mean-daily 5/15/30/60min RankIC；
- gradient norm；
- quantile calibration；
- checkpoint save/load parity。

### Gate D：完整训练

只有 pilot 在 validation 上产生稳定业务增益时，才运行：

`configs/experiments/zn_rank_e2_selected20_tod.json`

LoRA 和 dense F1/F1-MV 路线应放在 head-only 的输入价值被证明之后。

## 12. 复现路径

2025-08 bundle：

`data/zn-rank-selected100-1min/test-202508`

配置：

- `configs/experiments/zn_rank_e0_zero_shot_202508.json`
- `configs/experiments/zn_rank_e1_zero_shot_202508.json`
- `configs/experiments/zn_rank_e2_zero_shot_202508.json`

输出：

- `outputs/zn-rank-e0-zero-shot-202508/evaluation-test`
- `outputs/zn-rank-e1-zero-shot-202508/evaluation-test`
- `outputs/zn-rank-e2-zero-shot-202508/evaluation-test`

执行命令示例：

```bash
conda run --no-capture-output -n timesfm-ft \
  timesfm-eval \
  --config configs/experiments/zn_rank_e2_zero_shot_202508.json \
  --split test \
  --batch-size 32 \
  --device cuda
```
