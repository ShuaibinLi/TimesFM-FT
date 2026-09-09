# TimesFM-3 训练路线深度理解与业务微调设计

> 适用场景：1min intraday return + market features，TimesFM-3 作为 pretrained backbone。  
> 更新时间：2026-09。  
> 本文目的：把当前对 TimesFM-3 训练机制、公开 inference 实现、可能的内部 pretraining task construction，以及我们自己的 downstream fine-tuning 路线，整理成一份可复用的工程说明。

---

## 0. 先给结论

当前最重要的理解不是“TimesFM-3 的官方 pretraining loss 已经被完整还原”，而是：

1. **模型结构本身是完整且一致的**：输入 variate 与 raw output variate 是 slot 对齐的；每个 temporal token 经共享 backbone 后，输出未来 64 points × 9 quantiles。
2. **公开 PyTorch 代码是 inference-only port**，但保留了大量明显来自训练版 Flax/JAX 实现的接口和数据结构痕迹。
3. 公开代码中的 `get_output_patch_via_roll()` 明确将 shifted future patches 称为 “labels”，强烈支持“每个 temporal token 都可以作为 forecast origin”的训练范式。
4. 因此，TimesFM-3 原始训练很可能不是只对最终 horizon 做监督，而是更接近 **dense shifted forecasting**：多个 temporal anchors 同时预测其后 64 points。
5. CPM 更应该理解为一种 **task construction / mask distribution**，而不是另一个模型；inference 的 suffix masking 是这类训练任务的一个特殊边界条件。
6. 对我们的 1min 业务微调，最值得验证的主路线不是“猜官方 loss 并复刻”，而是：

```text
F0-final  : final-anchor return Pinball
F0-all    : all-eligible-anchor return Pinball
F1        : F0-all + final-anchor cumulative Huber @ 5/15/30/60min
F1-MV     : F1 + 少量 selected past-only auxiliary forecasting
```

其中 **F1 是当前最值得作为主候选的版本**，但 F0-final 必须保留为 deployment-consistent control。

---

## 1. 已确认事实、强推断与未知项必须分开

后续讨论始终使用三档证据等级：

| 等级 | 含义 | 本文写法 |
|---|---|---|
| **Confirmed** | 公开代码 / 官方 config / 官方文档可直接支持 | “已确认” |
| **Strong inference** | 多处实现痕迹一致，最合理解释高度集中 | “强推断 / 很可能” |
| **Unknown** | 公开实现不足以判断 | “未公开 / 不能确认” |

这一区分非常重要。我们的工程实现可以利用强推断，但不能把推断写成官方事实。

---

## 2. 已确认的 TimesFM-3 基础 topology

### 2.1 输入和输出

官方当前 PyTorch 模型的关键参数：

```text
input_patch_len  = 32
output_patch_len = 64
num_quantiles    = 9
quantiles        = 0.1 ... 0.9
model_dims       = 1280
num_layers       = 20
num_heads        = 16
use_variate_attention = true
```

原始连续序列在时间维上按 32-point patchify。对 1min 数据：

```text
input patch  = 32 min
output patch = 64 min
```

对每个 variate 和每个 temporal token，output head：

```text
1280 -> 64 * 9
```

因此底层输出 topology 是：

```text
[B, V, N, 64, 9]
```

这里的 `V` 包含 target、past-only、past-future 三类输入 variates；高层 Forecaster 最终只保留 target rows，不代表 backbone 没有为其它 variates 产生 raw logits。

### 2.2 variate slot 对齐

输入拼接顺序为：

```text
[target rows | past-only rows | past-future rows]
```

共享 backbone 不会重新打乱 variate axis。可以把 raw semantics 理解成：

```text
input variate j -> hidden variate slot j -> raw output variate j
```

因此模型并不需要额外学习 “哪个 output 对应哪个 target”。这个对应关系在 tensor layout 中已经固定。

### 2.3 Temporal + Variate mixing

每层 MixingTransformer 的结构是：

```text
Temporal causal attention
    -> Variate full attention at same temporal patch
    -> FFN
```

所以模型既可以学习单个序列的时间延续，也可以在同一 temporal patch 上做跨变量信息融合。

---

## 3. 公开 PyTorch 为什么看起来像“训练版的影子”

公开 PyTorch 文件明确写着：

```text
TimesFM3 PyTorch model (inference only)
forward(): equivalent to Flax __call__
```

这意味着它更像是 **内部 Flax 模型的 inference port**，而不是重新设计的一套独立 inference architecture。

更有价值的是，早期公开测试里保留了一个名为：

```python
test_forward_pass_training_dict
```

的测试，输入字段包括：

```text
values
masks
patch_segment_ids
patch_positions
patch_is_target
patch_is_past_only
patch_is_past_future_covariate
```

这套 schema 明显比当前高层 Forecaster 的：

```text
target
past_only_covariates
past_future_covariates
```

更加接近 pretraining dataloader / packed training batch。

### 3.1 这能支持什么结论

**已确认**：公开 Torch forward 仍然能够消费 patched tensors 与 role/mask metadata。  
**强推断**：内部训练版本很可能直接使用类似 `[B,V,N,P] + metadata` 的 batch schema。  
**不能确认**：公开仓库没有给出完整 dataloader，因此不能确定每个字段在真实 pretraining pipeline 中的全部含义和采样规则。

---

## 4. `get_output_patch_via_roll()` 是训练机制最关键的线索之一

公开 util 中的函数：

```python
get_output_patch_via_roll(x, rolls)
```

其 docstring 直接说它会：

```text
Creates labels of output_patch length by rolling the patched inputs.
```

对 TimesFM-3：

```text
P_in  = 32
P_out = 64
rolls = 2
```

设完整时间序列被切成：

```text
P0 P1 P2 P3 P4 P5 ...
```

则 shifted label topology 是：

```text
P0 -> P1 + P2
P1 -> P2 + P3
P2 -> P3 + P4
P3 -> P4 + P5
...
```

即：

\[
h_j \rightarrow y_{j+1:j+2}
\]

或者按原始点级别：

\[
h_j \rightarrow \text{next 64 points}
\]

### 4.1 为什么这是一个很强的训练线索

如果训练只关心“最后一个 horizon”，没有必要为每个 patch token 构造 shifted next-64 labels。这个工具的设计天然适合：

\[
\boxed{\text{每个 eligible temporal token 都作为 forecast origin}}
\]

因此，**all-token / dense-anchor forecasting 是目前最强的训练机制推断之一**。

但仍要注意：公开代码没有提供 loss loop，因此“哪些 token 最终真的进入 loss”仍不是 100% 可确认。

---

## 5. 训练与 inference 的 patch 对齐是同一个 topology

在当前 decode 逻辑中：

```text
最后一个真实 context token -> future points 1..64
第一个 future masked token -> future points 33..96
第二个 future masked token -> future points 65..128
```

相邻 output patches：

```text
length  = 64
stride  = 32
overlap = 32
```

然后通过 stitching 线性混合重叠部分。

这与 shifted training label：

```text
token j -> next 64 points
```

是完全一致的 topology。

因此可以把模型理解为：

```text
训练：在多个真实/部分 masked token 上学习 next-64 forecasting
推理：把最后一个真实 token 之后扩展成 future masked token，再沿同一 topology 解码
```

这里的差异主要在 **task construction / masking**，不是 backbone 结构变化。

---

## 6. CPM 的正确理解：不是 AR value rollout，而是 causal latent propagation

TimesFM-3 inference 时会先创建 future target / past-only masked slots。它不是：

```text
预测 y1 -> 把 y1 填回输入 -> 再预测 y2
```

而是：

```text
history | MASK | MASK | MASK | ...
```

一次 full-sequence forward。

Temporal attention 仍然 causal，所以第 k 个 future slot 可以看到：

- 真实历史；
- 更早 future slot 的上一层 hidden state；
- 但看不到 future ground truth。

因此更准确的术语是：

```text
causal latent-state propagation
```

而不是 autoregressive value generation。

### 6.1 信息论上的一个重要点

deterministic AR rollout 中：

\[
\hat y_1=f(X),\qquad
\hat y_2=g(X,\hat y_1)=g'(X)
\]

预测值 \(\hat y_1\) 并没有带来新的观测信息，只是提供了结构化中间状态。TimesFM-3 的 latent propagation 也是在显式利用这类结构性依赖。

真正的新信息只会在未来真实 observation 到达后产生。对于交易系统，这意味着线上可以每分钟重新 forecast 一次，而不必指望一个 64min rollout 内自行产生“新信息”。

---

## 7. CPM 训练方式：哪些是事实，哪些是外部类比

公开 TimesFM-3 PyTorch 代码中已经有：

```text
patch_cpm_mask
```

并在 preprocessing 中把目标类 variates 的对应 patch 置为 masked。

**已确认**：TimesFM-3 inference 使用 CPM 风格的 future masking。  
**未公开确认**：Google 没有公开完整 TimesFM-3 pretraining loop，因此不能直接证明“训练中随机 contiguous mask 的采样分布”。

TiRex 的公开论文给出了一个非常接近的训练范式：

- pretraining 中随机 mask 完整、连续 patches；
- target 仍然是 shifted future；
- information flow 保持 causal；
- loss 对多个 output tokens 计算。

这对 TimesFM-3 是一个 **非常有价值的外部机制参照**，但不能写成 TimesFM-3 的官方实现细节。

我们对 TimesFM-3 的更稳妥表述应是：

\[
\boxed{\text{CPM-compatible dense shifted forecasting 很可能是其训练分布的一部分}}
\]

而不是：

```text
TimesFM-3 一定完全照 TiRex 的 CPM recipe 训练
```

---

## 8. `segment_ids` 暗示可能存在 sequence packing

TimesFM-3 Transformer 支持：

```text
segment_ids
segment_pos
```

并构造 segment mask，禁止不同 segment 之间互相 attention。

同时，running RevIN statistics 也支持 segment-aware reset。

因此在一个 physical tensor 中可以表达：

```text
| series A | series B | series C |
| seg 0    | seg 1    | seg 2    |
```

且：

```text
A 不 attention B
B 不 attention C
RevIN 在 segment boundary reset
position 可按 segment 重新组织
```

### 8.1 当前判断

**已确认**：模型 primitives 支持 segmented full-sequence training。  
**强推断**：这非常适合 foundation-model pretraining 中的 sequence packing，且保留这些机制的工程动机很强。  
**未确认**：公开仓库没有训练 dataloader，不能断言 Google 的真实数据 pipeline 一定启用了 packing 或具体 packing ratio。

---

## 9. 三类 variates 的训练角色：真正未知的是 supervision set Ω

当前 inference 中：

```text
target      -> future unknown / masked
past-only   -> future unknown / masked
past-future -> future known / visible
```

raw head 对所有 variate slots 都输出 logits。

因此最关键的未知不是“网络能不能预测 past-only”，而是：

\[
\boxed{\Omega = \text{哪些 variate × token × horizon × quantile coordinates 进入训练 loss}}
\]

目前可以给出的置信度判断：

| 推断 | 置信度 |
|---|---:|
| 9 quantiles 参与监督 | 很高 |
| shifted next-64 labels | 很高 |
| 不只最后 temporal token 有监督 | 高 |
| target rows 有 forecasting loss | 极高 |
| past-only 可能存在 auxiliary forecasting loss | 中等~中高 |
| past-future 应主要作为 conditional input，而非 forecast target | 高 |
| role 在 pretraining task 中可能动态采样 | 中等 |
| 不同 role 可能有不同 loss weight | 中等 |

公开代码不足以确定：

```text
w_target
w_past_only
w_past_future
```

以及 role 是否按 dataset、series、window 或 patch 粒度采样。

---

## 10. 官方 pretraining loss 最可能是什么形态

旧 TimesFM / TimesFM-2.5 的公开训练实现保留了 point MSE + quantile loss 的 lineage，但 TimesFM-3 的公开 output head 当前只有 9 quantiles，没有单独 point/mean head。

因此 TimesFM-3 的核心 objective 最自然地收敛为某种 masked quantile objective：

\[
L_{base}
=
\frac{1}{|\Omega|}
\sum_{(b,v,j,h,q)\in\Omega}
\rho_q\left(y_{bvjh}-\hat y_{bvjhq}\right)
\]

其中：

- \(j\)：forecast origin / temporal token；
- \(h\in[1,64]\)：output patch 内 point；
- \(q\in\{0.1,...,0.9\}\)；
- \(\Omega\)：有效监督集合。

### 10.1 这里仍然不能确认的部分

不能从公开代码确认：

- loss 是所有 quantiles 等权还是加权；
- 是否单独加 median / mean loss；
- past-only auxiliary 的比例；
- CPM-masked token 和普通 token 的 loss 是否权重不同；
- 是否使用 curriculum / mixed task weighting。

所以我们自己的业务 fine-tune 不应建立在这些未知超参数上。

---

## 11. 对我们 1min intraday 业务的关键启发：一个样本可以提供多个 forecast anchors

当前业务配置：

```text
frequency = 1min
C_min     = 64
C_max     = 192
H         = 64
P_in      = 32
P_out     = 64
```

若某训练样本使用：

```text
C = 192min
```

则 context 有 6 个 temporal patches。再附加 64min ground-truth horizon，相当于完整序列中有 8 个 32min patches：

```text
context: P0 P1 P2 P3 P4 P5
future : P6 P7
```

对 context 中的每一个 anchor token，都可以构造 shifted next-64 label：

```text
P0 -> P1 P2
P1 -> P2 P3
P2 -> P3 P4
P3 -> P4 P5
P4 -> P5 P6
P5 -> P6 P7
```

这就是 dense-anchor supervision 的核心价值。

### 11.1 但不能简单把 6 个 anchor 全部当成等价业务样本

如果业务规定：

```text
C_min = 64min true history
```

那么 P0 在这个 sample 中只拥有 32min visible history。为了让训练分布和线上规则一致，建议定义：

\[
\mathcal J_{eligible}
=
\{j:\text{real visible history at }j\ge C_{min},\ 
\text{next 64min labels valid and same-session}\}
\]

对 C=192：

```text
P1, P2, P3, P4, P5
```

是严格满足 C_min=64 的 5 个 dense anchors。

如果要更贴近 foundation pretraining，可以额外做一个 “all-valid-token” ablation，让 P0 也参与，但不要把它与严格业务训练混为一谈。

---

## 12. 为什么 dense-anchor supervision 可以缓解“训练数据少”

原来的 final-anchor 训练：

```text
1 window -> 1 forecast origin
```

dense-anchor：

```text
1 window -> K eligible forecast origins
```

当 C=192 且 C_min=64 时，常见 K≈5；若允许更短 early anchors，K≈6。

这不会把高度重叠的 anchors 变成真正独立样本，因此不能夸大有效样本量，但它会显著增加：

- 每次 forward 的监督密度；
- 不同 intraday anchor 的梯度利用率；
- 对 shared forecasting geometry 的约束。

因此我们应把它理解为：

\[
\boxed{\text{提升 label efficiency，不是凭空增加独立市场样本}}
\]

---

## 13. 我们自己的 fine-tuning 目标：F0-final / F0-all / F1 / F1-MV

### 13.1 F0-final：最干净的 deployment-consistent baseline

只在最终业务 anchor 上，对未来 64 个 1min return 的 9 quantiles 做 Pinball：

\[
L_{final}
=
\frac1{64\times9}
\sum_{h=1}^{64}\sum_q
\rho_q(y_{T+h}-\hat y_{T,h,q})
\]

优点：

- 与真实线上使用完全一致；
- 实现最简单；
- 最适合作为所有后续训练方法的 control。

缺点：

- 一个 window 只用一个 anchor；
- 没有充分利用 decoder-only shifted forecasting topology。

### 13.2 F0-all：dense-anchor Pinball

对所有 eligible anchors：

\[
L_{all}
=
\frac1{|\mathcal J|}
\sum_{j\in\mathcal J}
\frac1{64\times9}
\sum_{h,q}
\rho_q(y_{j+h}-\hat y_{j,h,q})
\]

其中 \(\mathcal J\) 必须满足：

```text
足够真实历史
next-64 label 不跨 session
label 未被 halt / invalid mask
future-known features 对该历史 anchor 当时也确实已知
```

这应该成为我们最重要的 forecasting ablation。

### 13.3 F1：当前主推荐候选

保留 dense generic forecasting，同时只在 final decision anchor 加业务累计约束：

\[
L_{F1}
=
L_{all}
+
\lambda_{cum}L_{cum,final}
\]

首版：

```text
lambda_cum = 0.3
```

对 q50 path：

\[
\hat R_h=\sum_{i=1}^h \hat y_{T+i,0.5},\qquad
R_h=\sum_{i=1}^h y_{T+i}
\]

\[
h\in\{5,15,30,60\}
\]

用 training-only robust scale：

\[
L_{cum,final}
=
\frac14\sum_h
Huber\left(
\frac{\hat R_h-R_h}{s_h}
\right)
\]

为什么 business loss 只放 final anchor：

- dense Pinball 负责保持 foundation forecasting geometry；
- cumulative Huber 负责对齐真实交易 decision；
- 避免高度重叠的历史 anchors 把业务 loss 重复放大；
- 降低 business objective 对 pretrained representation 的破坏。

### 13.4 F1-MV：selected past-only auxiliary

仅在 F1 稳定后加入：

\[
L_{F1-MV}
=
L_{F1}
+
\lambda_{aux}L_{aux}
\]

首版：

```text
lambda_aux = 0.05
```

只选少数稳定、相对可预测的状态量：

```text
realized volatility
spread / liquidity state
volume state
```

不建议一开始对所有 OFI / imbalance / noisy microstructure features 做 auxiliary forecasting。

`past-future` 不做 forecast supervision，因为其 future 已知且已经作为 conditional input。

---

## 14. 一个非常关键的工程区别：`decode()` 不能直接拿来训练

当前公开 `decode()` 带：

```python
@torch.no_grad()
```

因此业务 fine-tune 不能直接：

```python
loss(model.decode(...))
loss.backward()
```

首版训练必须走底层 differentiable `forward()`，自己构造：

```text
values
masks
patch_is_target
patch_cpm_mask
```

并自己抽取 forecast anchors / labels。

### 14.1 更重要的 caveat：公开 Torch `forward()` 是 inference-specialized

当前代码对 fully-masked patch 使用了 leading-mask-only 的 `cumprod` 逻辑，并在注释里明确说明这是 inference 时的行为；同时注释又暗示内部 Flax `__call__` 在 training 下存在不同分支。

因此：

\[
\boxed{\text{如果要复刻原始 pretraining-like CPM training，不能默认公开 Torch forward 就是完整训练语义。}}
\]

这带来两条工程路线：

### 路线 A：deployment-consistent fine-tune

用当前 inference-shaped graph 做 differentiable training：

```text
真实 context + suffix masked horizon
```

先跑 F0-final，确保 train / deploy graph 尽量一致。

### 路线 B：pretraining-like dense training

实现明确的 full-sequence training path：

- causal attention；
- shifted labels；
- eligible anchor mask；
- 必要时 random CPM；
- 对 training/inference patch-mask semantics 做显式区分。

F0-all / F1 属于这一路线。

因此我们的开发顺序应该是：

```text
先让 F0-final 正确工作
-> 再实现 F0-all
-> 再加入 cumulative Huber
-> 最后才研究 random CPM / auxiliary roles
```

---

## 15. All-token training 的 leakage 审计

dense-anchor training 最大风险不是 temporal causal attention，而是 **feature availability semantics**。

对任意历史 anchor \(t_j\)，都必须问：

```text
这个 feature 在 t_j 当时，是否真的可获得？
```

### 15.1 Past-only

例如：

```text
OFI
spread
book imbalance
realized volatility
market return
```

它们未来未知，所以在该 anchor 之后不能作为 lookahead 输入。

TimesFM 的 target-like mask 正是为这类 future-unknown variates 服务。

### 15.2 Past-future

例如：

```text
sin/cos time-of-day
time-to-close
session phase
预先公布的 auction / rebalance window
```

它们在每个 anchor 当时都可以知道未来值，才可以作为 full context+future covariates。

特别警惕：

```text
future realized OFI
future spread
future realized vol
future volume
```

这些绝不能因为“训练数据里已经存在”就放入 past-future。

---

## 16. Dynamic context 与 all-token loss 如何兼容

线上 context：

```text
C_t = min(C_max, available intraday history)
C_min <= C_t <= C_max
```

不需要固定补到 C_max；只左 pad 到下一个 32-point boundary。

训练时可以保留同一规则，但 all-token supervision 要额外维护：

```text
anchor_valid_mask[j]
```

至少检查：

```text
real_history_at_anchor >= C_min
next_64_label_inside_same_session
label validity
feature availability
```

这样可以避免“最终 sample 有 192min context，但最早 token 只看到 32min”的隐含分布偏差。

---

## 17. 推荐的训练阶段

### Stage 0：数据 / semantics 冻结

冻结：

```text
1min return 定义
price source
timestamp semantics
session boundary
latency convention
feature availability
missing / halt policy
```

### Stage 1：Zero-shot + classical baseline

至少：

```text
zero-return
last-return / simple momentum
Ridge
LightGBM
TimesFM zero-shot return-only
TimesFM + past-only
TimesFM + past-future
```

### Stage 2：F0-final

目标：验证 end-to-end differentiable fine-tuning pipeline 与 deployment graph。

### Stage 3：F0-all

目标：验证 dense shifted supervision 是否带来：

```text
更稳 validation loss
更高 IC / RankIC
更快收敛
更少 overfit
```

### Stage 4：F1

加入 final-anchor 5/15/30/60min cumulative Huber。

### Stage 5：F1-MV

只在 F1 已稳定优于 F0-all 时加入 selected auxiliary past-only。

### Stage 6：CPM / role-sampling ablation

如果业务数据足够且需要更接近 pretraining distribution，再研究：

```text
random contiguous patch masking
role dropout / role reassignment
selected variate masking
```

这一步不是 v1 必做项。

---

## 18. 第一版推荐实验矩阵

| ID | Training graph | Forecast loss | Business loss | Aux | 目的 |
|---|---|---|---|---|---|
| **T0** | final suffix mask | final anchor Pinball | none | none | 必跑 control |
| **T1** | full causal sequence | all-eligible-anchor Pinball | none | none | 验证 dense supervision |
| **T2** | full causal sequence | all-eligible-anchor Pinball | 0.3 cumulative Huber final-only | none | **主候选** |
| **T3** | same as T2 | same | same | 0.05 selected past-only | multi-task ablation |
| **T4** | T2 + random CPM | same | same | none | pretraining-like mask ablation |

对应命名：

```text
T0 = F0-final
T1 = F0-all
T2 = F1
T3 = F1-MV
```

---

## 19. Checkpoint selection 不能只看 total validation loss

至少记录：

```text
IC / RankIC @ 5,15,30,60min
sign accuracy
prediction decile monotonicity
q10-q90 coverage
quantile crossing rate
cost-adjusted PnL
turnover
max drawdown
context bucket performance
open / midday / close performance
high-vol / low-vol regime performance
```

特别是：

```text
T3 total loss < T2
```

并不意味着 T3 更好。如果 return IC / PnL 没有提升，auxiliary objective 就没有业务价值。

---

## 20. 当前推荐的 v1 主路线

综合模型 lineage、公开代码和业务目标，当前最合理的执行顺序：

```text
1. Zero-shot / classical baselines
2. F0-final：保证训练链路正确
3. F0-all：验证 dense-anchor supervision
4. F1：all-anchor Pinball + final cumulative Huber
5. F1-MV：少量 auxiliary past-only
6. random CPM / role-sampling：最后才做
```

当前主候选 objective：

\[
\boxed{
L
=
L_{Pinball,all\ eligible\ anchors}
+
0.3L_{CumHuber,final\ anchor}
}
\]

它同时保留两种性质：

```text
foundation-model-like dense forecasting geometry
+
business-specific final decision alignment
```

这是目前最值得投入的一条训练路线。

---

## 21. 当前仍然需要继续验证的问题

以下问题不要假装已经知道答案：

1. TimesFM-3 官方 pretraining 是否对 past-only rows 计算 forecasting loss？
2. 官方训练的 `Ω` 到底包含哪些 token / variate positions？
3. random CPM 的真实 mask-length / mask-ratio 分布是什么？
4. training branch 中 fully-masked patches 是否作为 K/V 被排除，还是保留 latent propagation？
5. role assignment 是固定 schema 还是动态 task sampling？
6. 是否存在额外 median / point objective？
7. sequence packing 在真实 pretraining 中到底如何使用？

这些都应该通过进一步源码历史、论文附录、作者 release 或实验 parity 继续验证，而不是在业务代码里硬编码成“官方事实”。

---

## 22. 参考实现与证据入口

### TimesFM-3 官方仓库

- `src/timesfm3/torch/model.py`
  - inference-only 标记
  - `forward(): equivalent to Flax __call__`
  - 32/64 patch topology
  - CPM preprocessing
  - output head
  - decode stitching
- `src/timesfm3/torch/util.py`
  - `get_output_patch_via_roll()`：shifted next-output labels
  - segment-aware running stats
  - stitching
- `src/timesfm3/torch/transformer.py`
  - causal temporal attention
  - variate attention
  - segment mask / segment positions
  - full-sequence / no-cache path
- `src/timesfm3/torch/model_test.py`
  - `test_forward_pass_training_dict`
  - `patch_segment_ids / patch_positions / role masks`
- official branch / commit history
  - `timesfm-3.0`
  - earliest functioning Torch TimesFM-3 commit around `0548f536...`

### 外部机制参照

- TiRex / Contiguous Patch Masking：用于理解“random contiguous patch masking + shifted causal forecasting + dense output-token loss”这一类训练范式。

> 注意：TiRex 只作为机制类比，不作为 TimesFM-3 官方训练 recipe 的证据。

---

## 23. 一句话版本

我们现在最合理的训练理解是：

\[
\boxed{
\text{TimesFM-3 很可能是一个对多个 temporal forecast origins 做 shifted quantile forecasting 的模型，}
\text{CPM 改变的是可见信息分布，而不是输出结构。}
}
\]

因此我们的业务 fine-tune 应该优先利用这种 dense forecasting geometry，再把真正交易相关的 cumulative objective 稀疏地放在 final decision anchor 上，而不是简单照搬一个未知的官方 pretraining loss。
