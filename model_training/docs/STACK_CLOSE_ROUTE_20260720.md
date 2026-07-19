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

正式代码 commit：

```text
52bb4d9
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

## 6. 正式训练

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
