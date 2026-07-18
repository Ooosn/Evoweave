# Puppeteer 拓扑长尾因果诊断

状态：2026-07-18 已在 HGC 两张 H100 上完成固定协议评测、逐 token
干预、全训练集 topology 审计和 valid-common 补充评测。

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

## 下一项受控修复

不得重复此前的 joint-count-bin sampler probe。该 probe 只平衡长度分桶，
不能平衡同一长度内 4,930 次对 1 次的 topology family skew。

正式启动前对 15,541 个训练行完成了 exact-topology sampler 审计：

- 完整 topology family 共 `2,890` 个；
- 单例 family 共 `2,143` 个，占 family 的 `74.15%`，自然采样只获得
  `13.79%` 的行概率；
- 频率 `>=100` 的 family 只有 9 个，占 family 的 `0.31%`，却获得自然采样
  `58.60%` 的行概率；
- mixture alpha `0.75` 时，单例/`2..9`/`10..99`/`>=100` family 的期望
  行概率分别为 `59.06%/20.11%/5.95%/14.88%`；
- 在 300 step、effective batch 48 的 14,400 次暴露中，高频 family 仍有约
  2,143 次暴露，因此该设置不是完全删除 common family。

下一项只允许：

1. 新增 natural/topology-family-uniform mixture sampler；
2. sampler 以完整 `target_parents` 签名定义 rig family；
3. 使用 sequence-mean CE，避免短 skeleton 每行因 token 少再次被降权；
4. 第一轮不加 termination auxiliary loss，单独判断 topology-frequency
   修复是否改善长尾坐标和 topology；
5. 同时评测 heldout-52 与 valid-common60，且按 topology frequency 分层；
6. 52-joint 高频 family 不得明显退化，长尾 coordinate NLL、count、J2J 和 F1
   必须同时改善，不能只报告 EOS 或 teacher-forcing accuracy。

短程 checkpoint fine-tune 只能作为因果 probe，不能直接替代正式 baseline。
如果短程 probe 显示方向成立，正式结论必须来自同预算、从干净初始化开始的训练。

本次探针固定为 300 optimizer steps，每 100 step 保存一次；从
`checkpoint_sample_80000.pt` 严格加载完整模型，四组峰值学习率都为 `2e-5`，
OneCycle 重新开始，optimizer/scheduler 不从旧 checkpoint 恢复。除
topology-family mixture 与 `sequence_mean` 外，模型和数据契约保持不变。

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
```
