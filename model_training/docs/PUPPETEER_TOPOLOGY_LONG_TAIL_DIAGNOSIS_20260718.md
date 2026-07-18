# Puppeteer 拓扑长尾因果诊断

状态：2026-07-18 已在 HGC 两张 H100 上完成固定协议评测、逐 token
干预、全训练集 topology 审计、valid-common 补充评测、300-step exact-topology
重采样探针和 self-prefix rollout 审计。

## 结论

当前 `identity-1024 + pre-LN + joint-token` Puppeteer 模型没有发生输入/GT
错配，也没有忽略 mesh 或当前 pose。它真正学到的是少数高频 rig family 的
模板选择与姿态调整，没有学成长尾 topology 上普适的 condition-to-skeleton
映射。

该问题由三个因素共同形成：

1. 训练集按资产行采样，但大量资产共享完全相同的 parent tree。最主要的
   52-joint topology 单独出现 4,930 次。
2. 正式训练使用 token-mean CE。长骨架每行产生更多 token，因此高频、长骨架
   family 同时从行数和 token 数两方面主导梯度。
3. 自回归 decoder 在低 joint 边界几乎总看到 `continue`。当 condition-to-target
   映射不够强时，模型沿高频 topology 继续生成，最后停在 52、100、101 等
   训练长度 hazard 的尖峰。

因此，Puppeteer common 样本上的成功不能解释为“模型已经学会生成任意骨架”。
它能把未见过的 mesh/pose 映射到见过很多次的 rig family，但对低频或未见过的
rig family 明显失败。

## 固定评测协议

本轮使用：

- checkpoint：
  `/home/wangyy/evorig/outputs/puppeteer_identity1024_preln_hgc2h100_full_20260715/checkpoint_sample_80000.pt`
- 固定 heldout-52：
  `/home/wangyy/evorig/outputs/puppeteer_identity1024_preln_hgc2h100_full_20260715/length_balance_probe/heldout_52.jsonl`
- query pose seed：`101`
- surface/FPS seed：`101`
- 数据契约：rootless-v3、无 synthetic root、无 tail target、joint 0 为唯一真实根。

flat UniRig 与 Puppeteer 的原始 matched generation 已逐行确认 path、query
frame、selected frames、query center/scale 和 target joint count 一致。

heldout-52 的三个名称只描述数据来源和 joint count，不描述 topology 频率：

- `train-low`：训练 split 中选出的 16 行 4--10 joint 样本；
- `train-common`：训练 split 中随机选出的 16 行，joint count 为 22--92；
- `valid-low`：验证 split 中选出的 20 行 4--10 joint 样本。

topology 频率是另一维统计。它以完整、有顺序的 `target_parents` tuple 为签名，
在 15,541 个 `<=101` joints 的训练行中计数，并分为 `0`（unseen）、`1`
（singleton）、`2..9`、`10..99` 和 `>=100`。不得再把 `train-low` 或
`valid-low` 直接称为“低频 topology”。

## 已排除原因

### 输入或 GT pose 错配

固定同一资产的 pose-A GT 和 GT prefix，只把 condition 换成 pose-B：

- train-low 前十 joint 坐标 NLL 增加 `0.39~0.49`；
- valid-low 增加 `0.48~0.60`；
- train-common 增加 `2.85~2.93`。

正确 pose 条件在绝大多数 low 行上仍优于错误 pose。模型确实读取当前 pose，
但在 common rig family 上形成的 pose-to-skeleton 映射远强于 low family。

### FPS 随机采样

固定资产、pose、GT，只更换 surface/FPS seed：

- 坐标 NLL 平均绝对变化约 `0.02~0.04`；
- 同一测试中，换错 pose 的变化为 low `0.4~0.6`、common `2.9`。

FPS 会造成小幅预测扰动，但不是 low 系统性失败的主因。

### rootless joint 0 单点表示

GT-prefix 下逐 joint 统计：

- train-low joint 0 coordinate accuracy 为 `0.396`；
- valid-low joint 0 为 `0.400`；
- 后续 joint 多数更差。

自由生成首次错误落在 joint 0，是因为序列从 joint 0 开始且单轴准确率不足，
不是 joint 0 独有的代码错误。

### 数据源、坐标越界或动作量

- TexVerse 与 Objaverse-XL low 样本都同样失败；
- heldout low 只有两行 target joint 超出 mesh bbox，误差与坐标半径或跨度
  没有正相关；
- low 组 motion 统计略低于 52-joint 组，但在 36 个 low 样本内部，高 motion
  与低 motion 的 GT-prefix coordinate NLL/accuracy 基本相同。

动作证据不足会加重自由生成的长度和几何误差，但不能解释正确 GT prefix 下
整类坐标映射仍然欠拟合。

## 训练集 rig-family 分布

对训练 manifest 中全部 15,541 个 Puppeteer 可用样本建立 parent topology
索引；几何多样性统计使用全部 396 个 `<=10` 样本及各 400 个对照样本。

| 分组 | 样本 | 完整 topology 种类 | 最大 topology 占比 | 前四 joint 几何跨资产 RMS |
|---|---:|---:|---:|---:|
| `<=10` | 396 | 142 | 11.6% | 0.3371 |
| `==52` | 400 抽样 | 9 | 95.8% | 0.1285 |
| `20..75, !=52` | 400 抽样 | 178 | 12.5% | 0.2078 |
| `76..101` | 400 抽样 | 205 | 13.0% | 0.2522 |

完整训练集中：

- 52-joint 行数为 5,122，其中同一个 topology 出现 4,930 次；
- 28-joint 行数为 1,412，最大 topology 出现 1,337 次；
- 34-joint 行数为 834，最大 topology 出现 786 次。

这三个 family 已占 Puppeteer 训练行的很大部分。

## 自由生成确实落回高频模板

heldout-52 的 train-common 16 行中：

- 13 行生成的 parent tree 与训练集某个 topology 完全一致；
- 所有生成 52 joints 的行都落在出现 4,930 次的 topology；
- 所有生成 28 joints 的行都落在出现 1,337 次的 topology；
- 所有生成 34 joints 的行都落在出现 786 次的 topology。

low 行多数先沿高频 parent 模式延伸。短输出有时正好落在训练集短 topology，
过长输出通常不是完整复制某一棵训练树，但与同长度训练 topology 的 parent
序列高度相似。

## EOS 长度 hazard

训练集中，在已经生成到 joint `n` 的条件下，经验停止率为：

| n | `count == n` | `count >= n` | 经验停止率 |
|---:|---:|---:|---:|
| 4 | 43 | 15,541 | 0.28% |
| 5 | 63 | 15,498 | 0.41% |
| 6 | 98 | 15,435 | 0.64% |
| 10 | 53 | 15,198 | 0.35% |
| 12 | 111 | 15,095 | 0.74% |
| 28 | 1,412 | 13,042 | 10.83% |
| 52 | 5,122 | 8,478 | 60.42% |
| 92 | 70 | 198 | 35.35% |
| 100 | 23 | 38 | 60.53% |
| 101 | 15 | 15 | 100% |

这解释了为什么 low condition 一旦没有提供足够强的停止证据，decoder 会继续到
52 或接近 101，而不是在 4–10 joints 停止。

## 未见资产上的频率效应

另建 60 行 valid 补充集：

- 20 个 52-joint；
- 20 个其他 20–75-joint；
- 20 个 76–101-joint。

结果：

| valid 分组 | coordinate accuracy | count MAE | J2J | topology F1 |
|---|---:|---:|---:|---:|
| 52-joint | 0.7343 | 0.85 | 0.00837 | 0.6985 |
| 其他 20–75 | 0.3952 | 16.00 | 0.02837 | 0.4472 |
| 76–101 | 0.3056 | 27.35 | 0.02843 | 0.2813 |

52-joint 的 20 行中有 19 行生成 52 joints。这证明模型能把未见资产映射到高频
rig family，不只是记住训练资产。

按目标 topology 在训练集中的出现次数重新分组：

| topology 训练频率 | 行数 | coordinate NLL | count MAE | J2J | F1 |
|---|---:|---:|---:|---:|---:|
| `>=100` | 31 | 0.910 | 3.29 | 0.0108 | 0.672 |
| `10..99` | 6 | 1.865 | 19.17 | 0.0146 | 0.326 |
| `2..9` | 9 | 2.345 | 28.78 | 0.0333 | 0.338 |
| `1` | 5 | 2.109 | 25.80 | 0.0380 | 0.309 |
| `0` | 9 | 2.502 | 31.00 | 0.0433 | 0.129 |

`log(1 + topology frequency)` 与质量的 Spearman：

- coordinate NLL：`-0.843`
- coordinate accuracy：`+0.875`
- count absolute error：`-0.763`
- J2J：`-0.728`
- topology F1：`+0.781`

控制 target joint count 后，partial Spearman 仍为：

- coordinate NLL：`-0.829`
- topology F1：`+0.741`

只看 20 个 valid 其他中等长度样本，控制 joint count 后仍分别为 `-0.752`
和 `+0.817`。因此该关系不是 joint 数混杂造成的。

## 与 flat UniRig 的关系

两条路线不是同一个故障：

- Puppeteer 随机 decoder 对长尾 rig family 的 GT-prefix 坐标和 topology
  已经欠拟合，随后被长度 hazard 放大；
- flat UniRig 使用静态骨架预训练 decoder，common topology 与几何明显更好，
  但 10 个失败行在 self-generated prefix 下 EOS 概率坍塌并 hitmax。

共同点是自然行分布、长序列 token 权重和 teacher-forcing prefix 都会强化高频
长模板；区别是 Puppeteer 在正确 GT prefix 下已经失败，而 flat 的主要 hitmax
机制发生在 self-prefix rollout。

flat UniRig 的 10 个 hitmax 按目标 topology 的真实训练频率分层为：

| topology 训练频率 | 样本数 | hitmax |
|---|---:|---:|
| `>=100` | 12 | 0 |
| `10..99` | 20 | 2 |
| `2..9` | 11 | 3 |
| `1` | 3 | 3 |
| `0` | 6 | 2 |

这说明频率与 flat hitmax 也有关，但 heldout-52 样本量不足以把它单独解释成
唯一原因。

## Exact-topology 重采样探针

对 15,541 个训练行完成的 sampler 审计为：

- 完整 topology family 共 `2,890` 个；
- 单例 family 共 `2,143` 个，占 family 的 `74.15%`，自然采样只获得
  `13.79%` 的行概率；
- 频率 `>=100` 的 family 只有 9 个，占 family 的 `0.31%`，却获得自然采样
  `58.60%` 的行概率；
- mixture alpha `0.75` 时，单例/`2..9`/`10..99`/`>=100` family 的期望
  行概率分别为 `59.06%/20.11%/5.95%/14.88%`；
- 在 300 step、effective batch 48 的 14,400 次暴露中，高频 family 仍有约
  2,143 次暴露，因此该设置不是完全删除 common family。

探针固定为 300 optimizer steps，每 100 step 保存一次；从
`checkpoint_sample_80000.pt` 严格加载完整模型，四组峰值学习率都为 `2e-5`，
OneCycle 重新开始，optimizer/scheduler 不从旧 checkpoint 恢复。除
topology-family mixture 与 `sequence_mean` 外，模型和数据契约保持不变。

结果没有通过验收：

| 集合 | checkpoint | count MAE | J2J | topology F1 |
|---|---|---:|---:|---:|
| heldout-52 | baseline | 43.5192 | 0.053189 | 0.202308 |
| heldout-52 | step 300 | 39.8846 | 0.053737 | 0.208580 |
| valid-common60 | baseline | - | 0.021722 | 0.475656 |
| valid-common60 | step 300 | - | 0.024891 | 0.401310 |

valid-common60 中 44/60 行 J2J 恶化，43/60 行 F1 恶化。未见 topology 的
GT-prefix coordinate NLL 从 `2.5024` 略降到 `2.4456`，但自由生成 J2J 从
`0.04329` 恶化到 `0.04894`，F1 从 `0.12945` 恶化到 `0.08294`。因此：

- topology frequency 是真实的质量预测变量和训练暴露问题；
- 但只改变采样和 token loss reduction 不足以修复 condition-to-skeleton
  映射，并会破坏 common family；
- 该 step-300 checkpoint 已否定，不能升级为 baseline，也不得据此启动全量重训。

## Self-prefix rollout 审计

对 heldout-52 的每个目标位置分别计算：

1. 使用完整 GT prefix 时的下一个 token；
2. 使用模型从 BOS 开始自行 greedy 生成的 prefix 时的同一目标 token。

baseline 共比较 2,827 个 coordinate、940 个 parent 和 45 个到达的 EOS
位置：

| token 角色 | GT-prefix accuracy | self-prefix accuracy | self-prefix NLL 增量 |
|---|---:|---:|---:|
| coordinate | 0.5281 | 0.4075 | +1.7582 |
| parent | 0.9149 | 0.8149 | +0.8388 |
| EOS | 0.4667 | 0.1556 | +0.9654 |

按 heldout joint-count 分组，coordinate accuracy 为：

| 子集 | GT prefix | self prefix |
|---|---:|---:|
| train-low | 0.2904 | 0.1683 |
| train-common | 0.6048 | 0.4789 |
| valid-low | 0.2980 | 0.2071 |

所以当前故障不是二选一：

- 稀有/少关节目标在正确 GT prefix 下已经欠拟合；
- 第一个错误出现后，self-prefix 又让 coordinate、parent 和 EOS 全部进一步
  恶化，最终形成过长高频模板。

step-300 重采样模型的整体 coordinate accuracy 为 `0.5190 -> 0.3837`，
parent 为 `0.9107 -> 0.7741`，没有修复 rollout。审计自产 prefix 与正式
generation 在 baseline 51/52 行、probe 50/52 行完全一致；其余差异仅为少数
相邻量化 token，结论不依赖另一套生成协议。

## Sibling 顺序审计

Pass1 exporter 的 sibling 排序注释写的是 geometry order，但实际 key 使用
parent head，导致 sibling 几何 key 相同，再由名称打破平局，因此当前
`target_parents` 隐含了 bone-name lexical order。这是一个真实的表示契约缺陷。

但全训练集无序树审计显示，它不是长尾的主来源：

- 当前有序 topology `2,890` 种、singleton `2,143` 种；
- 忽略 sibling order 后仍有 `2,710` 种、singleton `1,963` 种；
- 仅 180 个有序 singleton 因忽略顺序而合并，占训练行约 `1.16%`；
- 改成 rest-geometry order 反而得到 `2,972` 种 topology，且当前 pose 下按
  geometry 动态排序会随动作变化，不能直接作为修复。

因此不能在没有新稳定 canonical-order 契约前重写数据顺序，也不能把主要失败
归因于名称排序。

## 当前边界

不得继续重复 joint-count-bin、exact-topology sampler、sequence-mean、
termination auxiliary、root/joint-0 或 FPS probe。它们已经回答了各自问题。
下一步必须改变表示或训练目标，使 topology/长度决策真正由 condition 支配，
同时保留 flat UniRig 已有的骨架先验；在该设计被代码级审计前不启动新的正式
多卡训练。

## 证据文件

```text
/home/wangyy/evorig/outputs/matched_heldout52_20260718/diagnostics/
puppeteer_same_asset_pose_swap_seed101_202.json
puppeteer_same_asset_pose_swap_seed101_303.json
puppeteer_same_pose_surface_swap_seed101_202.json
puppeteer_same_pose_surface_swap_seed101_303.json
skeleton_target_diversity_train_seed20260718.json
topology_template_lookup_sample80000.json
valid_common60_seed20260718.jsonl
puppeteer_valid_common60_teacher_seed101.json
puppeteer_valid_common60_generation_seed101.json
puppeteer_valid_common60_topology_frequency.json
puppeteer_heldout52_topology_frequency.json
puppeteer_heldout52_parent_counterfactual_seed101.json
puppeteer_heldout52_pose_prefix_vs_condition_seed101_202.json
puppeteer_heldout52_self_prefix_seed101.json
sibling_order_audit_quick.json
topology_order_ambiguity_quick.json

/home/wangyy/evorig/outputs/matched_heldout52_20260718/topology_alpha075_seqmean_ft300/
heldout52_step300_generation.json
heldout52_step300_teacher.json
heldout52_step300_topology_frequency.json
valid_common60_step300_generation.json
valid_common60_step300_teacher.json
valid_common60_step300_topology_frequency.json
heldout52_step300_self_prefix_seed101.json
self_prefix_summary.json
```
