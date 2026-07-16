# 当前模型状态（唯一入口）

更新时间：2026-07-16

状态 ID：`model-puppeteer-20260716-001`

本文档是模型模块的唯一当前状态入口。任何 agent 在训练、评测或解释
Puppeteer 之前必须先读本文档。历史根因的完整证据见
`PUPPETEER_CONDITION_COLLAPSE_DIAGNOSIS_20260715.md`；本文只记录已经确认的
事实、当前可信产物、未解决问题和下一步允许执行的工作。

## 1. 任务与固定数据契约

Evoweave 使用动态 mesh 序列生成当前 query pose 对应的完整骨架树：

```text
rootless-v3 dynamic NPZ
-> 1024 个有顺序的 mesh/motion condition tokens
-> autoregressive skeleton decoder
-> 当前 pose 的 rootless skeleton
```

固定契约：

- 数据只使用最终 rootless-v3 manifest；训练不得自行扫描 NPZ 目录。
- target 没有 synthetic root，也没有 tail token。
- `joint 0` 是 rootless 后唯一真实树根，不是假定的物体原点。
- 训练随机选择 query pose，mesh query 与该 pose 的 skeleton target 必须一致。
- 当前有两条 baseline：flat UniRig 和 joint-token Puppeteer。
- Puppeteer baseline token 为 `(x, y, z, parent_index)`；不混入 oracle-prefix、
  parent-delta 或其他增强实验。

## 2. Flat UniRig baseline

Flat UniRig rootless-v3 baseline 已经完成训练和评测，不是待训练项目。

Westlake 参考运行目录：

```text
/ssdwork/liuhaohan/evoweave/outputs/dynamic_rig_runs/
rootless_flat_unirig_motion_fullft_20260707_hxr4gpu
```

当前缺口：HGC 尚无一份在固定 Puppeteer heldout-52 上、使用同一 pose、同一
归一化和同一 CD 实现的 UniRig 结果。因此在补齐该对照前，不能用 Puppeteer
单独的 CD 数值宣称“好”或“接近 UniRig”。

## 3. 已失败的旧 Puppeteer 实现

历史四卡任务 `evoweave_jointtoken_hxr4gpu_20260709_878622c` 同时包含两个独立
实现问题：

1. 随机初始化的 24 层 SkeletonOPT 错误使用 post-LN。受控实验中，1000 step
   后 token accuracy 只有 `0.051`、parent accuracy 为 `0.000`；改为 pre-LN
   后分别达到 `0.416` 和 `1.000`。post-LN 是“训练后仍近似随机模型”的直接
   原因。
2. learned-query projector 把 1024 个有空间身份的 condition token 压缩为
   257 个匿名槽位。即使 decoder 改成 pre-LN，不同 pose/asset 的 condition
   仍会坍塌为近似相同的 pooled prompt。

受控 2x2 实验的 conditioner 相对差异：

| condition 路线 | surface | 同资产不同 pose | 不同资产 | 结论 |
|---|---|---:|---:|---|
| cross-attention-257 | 可训练 | 0.0244 | 0.0509 | 严重坍塌 |
| cross-attention-257 | 冻结 | 0.0281 | 0.0855 | 仍然坍塌 |
| identity-1024 | 可训练 | 1.2587 | 1.4418 | 保留差异 |
| identity-1024 | 冻结 | 0.5709 | 0.8459 | 保留差异 |

另一个独立风险是长期固定全量 `1e-4`：旧 direct-1024 实验到 step 5000 后，
同 pose/跨资产差异分别降到 `0.00124/0.00476`。OneCycle 峰值到 `1e-4`
不会产生同样坍塌，因此正式 baseline 必须使用 OneCycle，不能恢复固定学习率。

## 4. 已修复并锁定的 Puppeteer contract

正式 Puppeteer baseline 必须同时满足：

- `query_tokens=1024`
- `cond_length=1024`
- `condition_projection=identity`
- `decoder_norm_style=pre`，resolved `do_layer_norm_before=true`
- joint-slot embedding 开启
- random query pose 开启
- OneCycle scheduler
- 全部 condition 和 decoder 参数收到梯度
- `--require-query-preserving-baseline-contract` 开启；不满足时启动前直接失败

真实 rootless-v3 preflight 已确认：

- condition shape 为 `[1, 1024, 1024]`；
- 随机 query 为 frame 8，不是固定 frame 0；
- mesh query 与 target query 最大误差为 `0`；
- 当前 target 与 frame-0 target 的 RMS 差异为 `0.12856`，排除固定 reset-pose GT；
- 旋转 query 后 condition relative L2 为 `0.58355`，初始化路径没有条件坍塌；
- teacher forcing 与逐 token generation 在相同 prefix 上的 max logit diff 为 `0`；
- surface、motion、joint-slot embedding、decoder blocks 和 decoder token path
  均有有限非零梯度；没有未分配的可训练参数。

## 5. 当前可信 Puppeteer 正式运行

HGC 输出目录：

```text
/home/wangyy/evorig/outputs/
puppeteer_identity1024_preln_hgc2h100_full_20260715
```

训练事实：

- `2 x H100`；
- Puppeteer decoder 随机初始化，Evoweave/UniRig conditioner 初始化；
- 随后 `660,557,312` 个参数全部训练，没有冻结；
- micro batch `3/GPU`，gradient accumulation `8`，effective batch `48`；
- OneCycle，四组峰值学习率均为 `1e-4`；
- `1667` optimizer steps，累计 `80,016` 次样本输入；
- 训练行数 `15,541`；按 Puppeteer 上限过滤了 `379` 个大于 101 joints 的样本；
- checkpoint：`sample_10000`、`sample_40000`、`sample_80000`、`best_val`、`final`。

该运行是当前 Puppeteer source of truth。旧坍塌任务、短 overfit 和后续 probe
都不能替代它，也不能混成当前 baseline 结论。

## 6. 正式运行的已知评测结果

已经保存多随机种子自由生成、成功/失败 montage、teacher-pose diagnosis 和
分层评测。固定 heldout-52 的当前结果：

| 子集 | 数量 | J2J | J2B | B2B | joint-count MAE |
|---|---:|---:|---:|---:|---:|
| train-low-joint | 16 | 0.0774 | 0.0641 | 0.0549 | 45.12 |
| train-random/common | 16 | 0.0141 | 0.0120 | 0.0114 | 7.31 |
| valid-low-joint | 20 | 0.0673 | 0.0581 | 0.0509 | 43.75 |

这些数值只证明 Puppeteer 内部存在明显分组差异。由于没有同协议 UniRig 对照，
不得把 common 组的 `J2J=0.0141` 描述为“好”或“可用”。

## 7. 已做但未被接受的后续 probe

以下实验都从正式 checkpoint 出发，并重新运行 heldout-52：

- control fine-tune 200 steps；
- joint-count balanced fine-tune 200 steps；
- mixture alpha `0.25/0.50`，各 300 steps；
- sequence-mean token loss 200 steps；
- sequence-mean + termination loss 200 steps；
- mixture alpha `0.50/0.75` + sequence-mean，各 200 steps；
- termination weight `0.10/0.25` 的短程探针。

这些实验能够改善 parent、EOS 或预测 joint count，但没有同步改善低关节几何。
例如 alpha `0.75` + sequence-mean 将 valid-low 的长度中位数改善到 6，J2J
却从 `0.0673` 恶化到约 `0.0709`。因此它们均不是新的可信 baseline。

2026-07-16 启动的 50% low-joint + 50% common 全参数实验已在 step 50 停止，
没有保存 checkpoint，也没有改变当前 source of truth。

## 8. 已确认的诊断事实

- GT-prefix teacher forcing 下，common 组 coordinate accuracy 约 `0.6046`，
  train-low/valid-low 只有约 `0.2656/0.2919`。失败在正确 prefix 下已经存在，
  不能只归因于自由生成累计误差。
- 固定 GT 与 GT prefix 后，正确 condition 的 coordinate NLL 为 `2.6749`，
  跨资产错配 condition 为 `4.4735`；16/16 个 low-joint 样本均是正确 condition
  更优。因此当前正式模型没有把 condition 完全忽略，也没有发生输入/GT 配错。
- heldout-52 的首次自由生成错误全部出现在 `joint 0` 的坐标 token，不是 parent
  或 EOS。
- low-joint rootless `joint 0` 的目标分布远宽于 common 组，尤其 z 轴；这是
  当前候选解释，不是已经证明的根因。

## 9. 当前仍未解决的问题

1. 尚未证明 `best_val/final/sample_80000` 中哪个自由生成最好。
2. 尚未建立同协议 UniRig heldout-52 对照，无法判断 Puppeteer 绝对质量。
3. low-joint 首关节坐标为什么泛化失败仍未最终定因；训练分布不足与首关节表示
   不适配尚未完成决定性区分。
4. 当前没有任何后续 probe 可以替代正式 Puppeteer checkpoint。

## 10. 继续工作的硬规则

- 现在不得从零重训 Puppeteer；先使用现有 `10k/40k/80k/best/final`。
- 任何“变好”结论必须来自同一 manifest、同一 pose、同一归一化、同一指标实现
  的 UniRig/Puppeteer 对照，并附自由生成可视化。
- teacher-forcing accuracy、单独 CD 数值或 joint count 改善都不能作为验收。
- 新改动必须先通过 query/target 对齐、prefix logits、梯度和小规模自由生成检查；
  未通过时不得启动正式多卡任务。
- 最终验收必须同时检查当前 pose 对齐、joint count、EOS/hitmax、parent tree、
  J2J/J2B/B2B 和 GT/Prediction/Overlay。

## 11. 下一项唯一允许的执行工作

取得 flat UniRig 的可信 checkpoint，在现有 heldout-52 上用完全相同的 pose 和
几何指标运行 UniRig、`sample_80000`、`best_val`、`final` 四方对照并生成 montage。
完成该对照前，不提交新的长训练任务，也不修改数据契约。
