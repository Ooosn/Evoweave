# Puppeteer 条件坍塌诊断

状态：2026-07-15 已在 HGC 单张 H100 上完成分层追踪、对照实验与因果旁路。

## 最终结论

此前 Puppeteer 路线会为不同 query pose，甚至不同 mesh，生成同一套完整但错误的骨架。根因不在 GT 读取、FPS、Puppeteer 预训练权重或训练/生成位置对齐，而在当前 Evoweave motion encoder 的联合训练方式：

1. 输入的 1024 个 surface tokens 原本包含清楚的 anchor、mesh 和 pose 差异。
2. 每层 `pose_inner` attention 允许一帧内所有 anchors 全连接。该分支天然产生一个会广播给所有 anchors 的全局 pooled 分量。
3. 纯 teacher-forcing CE 允许 decoder 从 GT 坐标前缀恢复 pose，因而没有要求 condition 保留 query geometry。
4. 联合训练把 `pose_inner` attention 的公共方向选择性放大。block 0 的 `out_proj` 对公共分量的实际增益从初始化时约 `0.57x` 变为约 `3.89x`，对局部分量只有约 `1.55x`。
5. 这个公共向量在 12 层 pre-LN 残差流中反复相加，MLP 再进一步放大。最终残差 RMS 约为 `687`，而 pose 差异绝对 RMS 仍只有约 `3`。
6. 最后的 LayerNorm 把巨大公共向量归一化，得到几乎相同的 1024 个 condition tokens。多数不同 pose 和不同 mesh 都被映射到同一个近常量 soft prompt。
7. teacher forcing 下，decoder 仍可依靠 GT 历史 token 降低损失；自由生成没有 GT 历史，只能从这个通用 prompt 生成一个模板姿态，再自洽地延续成整套错误骨架。

因此，这不是“模型学得不够好”，而是当前目标函数允许、当前 motion encoder 结构又放大的条件不可辨识性。继续增加训练步数会让坍塌更严重。

## 定义

- **query pose**：一次数据读取随机选择的目标帧。输入 mesh 和监督骨架来自同一帧。
- **surface tokens**：surface tokenizer 对 query/evidence meshes 产生的逐 anchor 连续向量。
- **condition tokens**：motion encoder 输出并送入 AR decoder 的 1024 个连续向量。
- **slot 公共分量**：一个样本的 1024 个 token 在 slot 维求均值后得到的向量。其 RMS 占总 RMS 的比例接近 1，表示各 slot 几乎相同。
- **teacher-forcing prefix**：预测当前位置时提供给 decoder 的 GT 历史 skeleton tokens。
- **condition swap**：保持同一个 GT prefix，只交换 pose A/B 的 condition，用来隔离 condition 对 logits 的影响。
- **prefix swap**：保持 condition 不变，只替换 GT prefix，用来测量 decoder 对 GT 历史的依赖。
- **残差旁路**：不改 checkpoint，只在推理时把指定 attention/MLP 的更新量乘 0，用来验证该支路是否是信息擦除的原因。

## 已排除项

- 输入 mesh 与 target skeleton 使用同一个随机 query frame，不存在固定读取 reset-pose GT。
- 两种 pose 的 query mesh、target joints 和离散 GT tokens 都实际发生变化。
- 强制两种 pose 使用完全相同的 vertex/face/barycentric/FPS references 后，坍塌仍存在，因此不是采样差异导致。
- teacher forcing 与逐 token generation 在相同 prefix 上的 logits 对齐，不是位置偏移或 token shift 错误。
- surface tokenizer 输出在坍塌前能区分 pose 和 mesh。
- 同一随机初始化 decoder 配冻结 conditioner 可以学习并生成 pose-sensitive 结果，因此不是必须依赖 Puppeteer 预训练权重。
- joint-slot/target-aware embedding 不是这次固定模板故障的根因。
- learned role/register tokens 不是主要劫持源。训练后 block 0 的 pose-inner anchor queries 约 `95.4%` attention mass 仍落在真实 anchors，register 约 `4.6%`。

## 信息在哪里第一次丢失

4 个训练资产、pose seeds 101/202 的分层中位数：

| 阶段 | 左侧 RMS | pose 相对 L2 |
|---|---:|---:|
| raw surface tokens | 1.689 | 1.311 |
| trainable step1000, block 0 | 6.349 | 0.534 |
| trainable step1000, block 11 | 166.325 | 0.0206 |
| trainable step5000, block 0 | 23.228 | 0.151 |
| trainable step5000, block 11 | 686.959 | 0.00536 |
| trainable step5000, final condition | 1.008 | 0.00505 |
| frozen step3000, block 11 | 2.455 | 1.230 |
| frozen step3000, final condition | 0.996 | 1.233 |

step1000 已经开始坍塌；step5000 时 block 0 就丢掉大部分相对差异。最终 LayerNorm 只把绝对尺度恢复到约 1，不会恢复已经被公共方向淹没的信息。

### block 0 子层

| 子层更新 | 冻结 RMS | 可训练 step5000 RMS | 可训练输出的 slot 公共占比 |
|---|---:|---:|---:|
| pose-inner attention delta | 0.134 | 3.10 | 0.977 |
| pose-inner MLP delta | 0.198 | 13.34 | 0.994 |
| anchor-temporal attention delta | 0.378 | 2.96 | 0.996 |
| anchor-temporal MLP delta | 0.195 | 5.93 | 0.9997 |

参数整体 RMS 只改变约 10%，所以不能把它描述成普通的权重数值爆炸。真正变化是投影矩阵与全局 pooled 特征方向对齐：

- `V projection` RMS 只增加约 `1.4x`；
- attention 加权后的 pre-out RMS 增加约 `4.1x`；
- `out_proj` 后 attention 输出 RMS 增加约 `26.5x`；
- `out_proj` 对公共分量的实际增益约为 `3.89x`，对局部分量约为 `1.55x`；
- 冻结初始化时两者都约为 `0.57x`。

## 不只是 pose 坍塌

step5000 的 4 个资产中：

- 同一资产不同 pose 的 condition 相对 L2 为 `0.0042, 0.0058, 0.5859, 0.0043`；
- 不同资产之间为 `0.0047` 到 `0.6637`；
- 资产 0、1、3 之间只有约 `0.005`，与它们各自的 pose 差异同量级；
- 资产 2 是唯一保持差异的离群样本。

最终 condition 的 slot 公共分量占比为 `0.999993`，slot-centered RMS 只有约 `0.0037`。这直接解释了为什么一个 mesh 会得到另一个 mesh 的完整骨架：多数 mesh 在进入 decoder 前已经变成同一个条件。

冻结 conditioner 的严格对照中：

- 同一资产不同 pose 的 condition 距离为 `1.21` 到 `1.26`；
- 不同资产距离为 `1.30` 到 `1.32`；
- 最终 slot 公共分量占比约 `0.45`，局部 anchor 结构仍存在。

两组使用同一 seed、同一 motion encoder 初始化、同一数据和同一随机 decoder；唯一主要变量是 motion encoder 是否接收梯度。

## 分支因果旁路

在坏 checkpoint 上不重新训练，只旁路残差更新：

| 保留的 motion 分支 | 同资产 pose 距离 | 不同资产距离 | slot 公共占比 |
|---|---:|---:|---:|
| 原模型全部分支 | 约 0.005 | 多数约 0.005 | 0.999993 |
| 只保留 pose-inner attention | 0.0104/0.0109 | 0.0127 | 0.999973 |
| 只保留 anchor-temporal attention | 1.342/1.397 | 1.382 | 0.24 到 0.31 |
| 只保留 MLP，旁路全部 attention | 0.419/0.386 | 0.389 | 约 0.96 |
| 全部 motion 更新旁路 | 1.27 到 1.33 | 1.35 到 1.37 | 约 0.35 |

所以首要擦除器已经定位为 `pose_inner` attention；MLP 会放大坍塌，但不是唯一来源。全旁路后，前两个明显运动样本的前 8 个 greedy tokens 也重新随 pose 变化，说明该支路与固定生成之间存在直接因果关系。坏 decoder 只在坍塌条件上训练过，因此旁路 checkpoint 本身不会立刻变成高质量最终模型。

## teacher forcing 为什么允许它发生

每个 joint 被序列化为 `(x, y, z, parent_index)`。训练时预测当前 token 会看到此前所有 GT tokens。除最早的少数坐标外，大部分位置都能从已经提供的同一 pose 坐标前缀恢复后续姿态。

在可训练 step5000 checkpoint 上：

- condition swap 的 argmax 变化率为 0%；
- prefix swap 会在明显运动样本上改变约 20% 到 37% 的 argmax；
- condition/prefix logit 影响比通常只有 `0.006` 到 `0.012`。

冻结 conditioner 后，condition/prefix 影响比恢复到约 `0.72` 到 `1.32`，condition swap 能改变 logits 和部分 argmax。decoder 并非结构上无法使用 condition；联合训练只是找到了更容易的 GT-prefix shortcut。

## 为什么 UniRig 路线没有同样退化

差异不是“用了哪个预训练权重”这么简单。UniRig 路线的静态 mesh-to-skeleton 条件语义已经被其训练路径固定；当前 Puppeteer 试验则让随机 decoder 与一个没有保真约束的 12 层 motion encoder 同时自由协同。在纯 CE 下，中间 condition 没有唯一语义，二者可以共同退化为“通用 soft prompt + GT prefix”。

冻结随机 conditioner 的对照仍能学习，说明 Puppeteer decoder 不依赖某个神奇预训练初始化。真正缺失的是联合训练时对 query geometry 和 condition 使用方式的约束。

## 修复要求

下一版不能只改学习率、增加训练步数或永久冻结 conditioner。需要同时阻断编码器擦除和 decoder 逃逸：

1. **query-preserving path**：最终 condition 必须包含一条 motion encoder 无法覆盖的 query surface token 路径。可以采用逐 anchor query skip，再以有界 gate 融入 motion context。
2. **pose-inner LayerScale**：`pose_inner` attention/MLP 的残差更新需要小尺度初始化和有界控制，防止全局 pooled 方向在 pre-LN 残差流中无限累积。
3. **matched-vs-swapped condition loss**：从同一序列取两个 query poses。固定 pose A 的同一个 GT prefix，分别输入 condition A/B，只在真实发生变化的 coordinate positions 上要求 matched condition 的目标似然高于 swapped condition。这样直接监督“condition 必须解释 pose”，不会把错误坐标前缀强绑到 GT tree。
4. **训练中硬审计**：持续记录 slot-centered RMS、同资产 paired-pose condition L2、不同资产 condition L2、condition-swap logits/argmax。任何一项重新接近坍塌值都应终止任务。
5. **顺序**：先证明 condition 保真和 condition usage，再讨论 oracle/model-prefix rollout。否则 rollout loss 只是在已经不可辨识的 condition 上增加另一层变量。

在上述四项至少完成 query-preserving path、condition-swap supervision 和在线审计前，不应再次提交“motion encoder 全量可训练 + 纯 teacher forcing”的正式任务。

## 证据位置

HGC checkpoints：

```text
/home/wangyy/evorig/diagnostics/puppeteer_hgc_identity1024_preln_motion_decoder_32asset_5000step_6dd6d30
/home/wangyy/evorig/diagnostics/puppeteer_hgc_identity1024_preln_frozen_conditioner_32asset_3000step_1bd9785
```

HGC causal traces：

```text
/home/wangyy/evorig/diagnostics/puppeteer_condition_stage_trace_20260715
```

诊断实现：

```text
model_training/analysis/diagnose_puppeteer_teacher_forcing.py
```
