# Stack-Close 骨架生成路线

## 1. 目标与隔离边界

这条路线用于验证三项组合改动：

1. 用 DFS `CLOSE` 序列替换 flat UniRig 的 `BRANCH + parent xyz` 表示，消除量化重复坐标造成的 parent 身份丢失。
2. 训练时随机打乱同一 parent 下的兄弟子树顺序，避免模型把词法顺序误当成唯一拓扑。
3. 对 autoregressive coordinate prefix 做小幅骨段感知扰动，训练模型从轻微错误前缀继续生成合法骨架。

它是独立路线，不修改现有 flat UniRig baseline 的 tokenizer、dataset、trainer、launcher 或 checkpoint。
代码只位于：

```text
model_training/rigweave/src/rigweave/stack_close/
model_training/rigweave/scripts/train_stack_close_dynamic_rig.py
model_training/rigweave/scripts/eval_stack_close_generation.py
model_training/jobs/run_stack_close_sibling_perturb_20260720.sh
```

## 2. 数据与初始化

唯一数据来源：

```text
train  /ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests/train_manifest.jsonl
valid  /ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests/valid_manifest.jsonl
```

行数必须严格为 `15903 / 857`。训练只读 manifest，不能扫描 NPZ 目录。

模型从官方 static UniRig skeleton checkpoint 初始化，不加载任何 Evoweave dynamic checkpoint：

```text
/ssdwork/liuhaohan/evorig/Evoweave/model_training/third_party_references/UniRig_hf/skeleton/articulation-xl_quantization_256/model.ckpt
```

surface tokenizer、motion encoder 和 UniRig autoregressive transformer 全量训练。

## 3. Stack-Close 表示

序列格式为：

```text
BOS, CLASS, root.xyz, child-subtree..., CLOSE, ..., CLOSE(root), EOS
```

- `CLOSE` 复用旧 UniRig `BRANCH` 的 token ID `256`，因此词表、embedding 和输出 head 形状不变。
- parent 由解析栈直接确定，不再通过 parent 坐标反查。
- 每个真实 joint 恰好出现一次，每个 joint 恰好对应一个 `CLOSE`。
- grammar 不允许空栈 `CLOSE`、第二个 root、root 尚未关闭时的 EOS，或 EOS 后附加 token。
- 验证使用 canonical sibling order；训练对每个 sibling set 随机排列。

2026-07-20 全量审查：

- `16760 / 16760` 样本 canonical 加两次随机兄弟顺序均精确 round-trip。
- parent identity failure 为 `0`。
- train 中 `15715 / 15903` 行实际产生过不同兄弟顺序；valid 为 `844 / 857`。
- 最大 target 长度 train `1003`、valid `827`。
- baseline 文件修改数为 `0`。

## 4. Prefix 扰动

标签始终保持干净。只有喂给 decoder 的 coordinate prefix 会被扰动。

每个被选 joint 的位移由两部分组成：

```text
delta = axial * bone_axis + radial * plane_normal
```

- axial 最大幅度：该 joint 对应骨段长度的 `5%`。
- radial 最大幅度：沿骨段法平面任意角度，最多为近似 mesh 出射距离的 `5%`。
- radial 出射距离使用已经完成 FPS 的 `1024` 个 query surface anchors 在 GPU 上近似，不做在线 Open3D raycast，不增加 CPU dataloader 开销。
- 每行扰动概率 `0.5`。
- 每行最多扰动 `4` 个 joint，且不超过 joint 总数的 `8%`。
- 正式训练前 `5000` 次样本暴露不扰动，随后在 `15000` 次样本暴露内线性升到完整强度。

256 个真实样本的离线标定表明，`5% axial + 5% radial` 会让约 `79.18%` 的被选 joint 至少改变一个 256-bin coordinate token；精确 Open3D raycast 约 `0.37s/sample`，因此不用于在线训练。

## 5. Preflight 结果

第一条路线的不可变正式训练源：

```text
/ssdwork/liuhaohan/evorig/run_sources/stack_close_79e6da4
commit 79e6da43fda8225831c7d1ff41669963f9d058c8
```

固定配置 one-step 检查：

- AR 参数：`305,736,704`
- motion 参数：`303,526,912`
- surface 参数：`54,119,936`
- optimizer 参数总数：`663,383,552`
- 模型 trainable 参数总数：`663,383,552`
- 未分配、重复或冻结后误加入 optimizer 的参数均为 `0`
- 单卡 micro-batch `3` 峰值显存约 `64.5GB`

32 个 train-manifest 样本、600 次样本暴露的短训练自由生成结果：

- 合法 detokenize：`32 / 32`
- 主动 EOS：`32 / 32`
- hitmax：`0 / 32`
- topology F1 mean：`0.590121`
- topology F1 median：`0.616482`
- J2J mean（31 个具有骨段的有效行）：`0.052682`

短训练的主要失败模式是过早 `CLOSE` 导致骨架偏短，不是常量坐标循环、模板坍塌或无法终止。该结果只用于证明路线可训练和自由生成 contract 成立，不是最终性能结论。

服务器证据：

```text
/ssdwork/liuhaohan/evoweave/outputs/stack_close_preflight_20260720/roundtrip.json
/ssdwork/liuhaohan/evoweave/outputs/stack_close_preflight_20260720/fixed_optimizer_one_step
/ssdwork/liuhaohan/evoweave/outputs/stack_close_preflight_20260720/overfit32_freegen
```

## 6. Condition Refresh 独立路线

Condition refresh 不是第一条路线的隐式开关，而是单独的模型 profile：

```text
model_training/rigweave/src/rigweave/stack_close_refresh/
model_training/rigweave/tests/test_condition_refresh.py
model_training/rigweave/scripts/audit_condition_refresh_contract.py
model_training/jobs/run_stack_close_condition_refresh_20260720.sh
```

它在 OPT 的第 `8 / 16 / 24` 层后，只对 skeleton hidden states 追加一次到原始
`1024` 个 motion-aware condition tokens 的 cross-attention：

```text
1024 hidden -> 256 bottleneck -> 8-head cross-attention -> 1024 hidden
```

每个 refresh adapter 的输出乘以独立的逐通道 `tanh` gate。gate 精确初始化为
零，因此初始模型与不带 refresh 的第一条路线逐 logit 完全相同；训练打开 gate
后，refresh projection 才开始接收非零梯度。condition prefix 本身不被改写。

三层 adapter 共增加 `3,166,464` 个参数。one-step optimizer 审查结果：

- AR 参数：`305,736,704`
- motion 参数：`303,526,912`
- surface 参数：`54,119,936`
- refresh 参数：`3,166,464`
- trainable 与 optimizer 参数总数均为 `666,550,016`
- 未分配、重复或冻结后误加入 optimizer 的参数均为 `0`
- 单卡 micro-batch `3` 峰值显存约 `59.56GB`

同一批 32 个 train 样本、同为 200 optimizer steps / 600 次样本暴露的匹配短训：

| 指标 | 无 refresh | 有 refresh |
| --- | ---: | ---: |
| final validation loss | 1.920504 | 1.920864 |
| validation token accuracy | 0.393437 | 0.432542 |
| topology F1 mean | 0.590121 | 0.568839 |
| topology F1 median | 0.616482 | 0.546575 |
| J2J mean | 0.052682 | 0.057209 |
| 合法 EOS | 32 / 32 | 32 / 32 |
| hitmax | 0 / 32 | 0 / 32 |
| median step time | 2.87990s | 2.90510s |

结论只到以下范围：

- condition refresh 的训练、generation cache 和 grammar contract 均成立，没有模板坍塌；
- 完整 trainer 的实测步耗时只增加约 `0.9%`；
- 600 次样本暴露下没有性能提升证据，F1 和 J2J 反而略差；
- 主要错误仍是过早 `CLOSE`；另有 1 行产生长分支过生成，但最终主动 EOS。

因此第二条路线值得作为独立正式对照训练，但不能在正式验证结果出来前写成已经优于
第一条路线。

服务器证据：

```text
/ssdwork/liuhaohan/evoweave/outputs/stack_close_refresh_preflight_20260720/contract.json
/ssdwork/liuhaohan/evoweave/outputs/stack_close_refresh_preflight_20260720/train32_step200
/ssdwork/liuhaohan/evoweave/outputs/stack_close_refresh_preflight_20260720/train32_step200_freegen
```

## 7. 正式训练

第一条正式任务固定为：

- `2 x A100 80GB`
- 非 `huangxiangru` 资源组
- micro-batch `3/GPU`
- gradient accumulation `8`
- effective batch `48`
- `1667` optimizer steps，约 `80016` 次样本暴露
- AdamW，weight decay `0.04`
- surface、motion、AR 三组 OneCycle peak LR 均为 `1e-4`
- 从 static UniRig 初始化，全量微调
- 不启用 condition refresh、oracle、explicit-tree、旧 recovery loss 或任何 fallback

第二条 condition-refresh 路线必须在另一个独立 profile 和 commit 中实现，并同样从 static UniRig 初始化。它不能覆盖第一条路线，也不能修改 flat UniRig baseline。

第二条正式任务沿用完全相同的数据、初始化、优化器、学习率、样本暴露数和
`2 x A100 80GB` / effective batch `48` 配置，唯一模型差异是本节定义的三层
zero-residual condition refresh。
