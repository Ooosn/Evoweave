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

Flat UniRig rootless-v3 在 Westlake 有一份历史训练结果，但其动态 checkpoint 当前不在
HGC，不能再把取得该 checkpoint 当作继续工作的前提。2026-07-17 已明确允许在 HGC
从官方静态 UniRig skeleton checkpoint 重新训练一份干净的同预算对照线。

Westlake 参考运行目录：

```text
/ssdwork/liuhaohan/evoweave/outputs/dynamic_rig_runs/
rootless_flat_unirig_motion_fullft_20260707_hxr4gpu
```

HGC 重训固定使用当前 rootless-v3 train/valid manifest、随机 query pose、完整
surface/motion/AR 全量训练、OneCycle、effective batch 48 和 80016 次样本暴露。
不得加载旧 Evoweave 动态 checkpoint。

该重训已于 2026-07-18 完成：

- 输出目录：
  `/home/wangyy/evorig/outputs/flat_unirig_hgc2h100_matched80k_20260717`
- `2 x H100`，`1667` optimizer steps，累计 `80,016` 次样本输入；
- `662,268,416` 个参数全部训练；
- `checkpoint_sample_80000.pt` 包含 model、optimizer 和 scheduler；
- step 1600 的验证集 CE 为 `1.165135`，EOS accuracy 为 `1.0`；
- 训练 CE 不能作为生成质量验收，正式结论只使用第 6 节的 matched generation。

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

## 6. 同协议 heldout-52 正式评测

2026-07-18 已完成 flat UniRig 与三个 Puppeteer checkpoint 的四方自由生成对照。
四份结果逐行硬校验了相同的 `path`、query frame、24 个 selected frames、
query center/scale 和 GT joint count；因此这里不存在 pose、归一化或样本错配。

结果目录：

```text
/home/wangyy/evorig/outputs/matched_heldout52_20260718
```

全量 52 行结果：

| 路线 | 可解析 | hitmax | joint-count MAE | J2J | J2B | B2B | topology F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat UniRig sample80000 | 42/52 | 10 | 6.0952 | 0.038961 | 0.027517 | 0.022120 | 0.639410 |
| Puppeteer sample80000 | 52/52 | 0 | 43.5192 | 0.053189 | 0.044258 | 0.038557 | 0.202308 |
| Puppeteer best_val | 52/52 | 0 | 44.0000 | 0.054615 | 0.046626 | 0.040644 | 0.197214 |
| Puppeteer final | 52/52 | 0 | 43.6154 | 0.052854 | 0.043918 | 0.038224 | 0.199103 |

flat UniRig 的几何和拓扑统计只在 42 个成功解析的样本上计算；另外 10 个样本
没有 EOS，达到 1400-token 上限后被 tokenizer 拒绝，整套输出不可用。不能用
42 个成功样本的均值掩盖这 10 个失败。

flat UniRig 与 Puppeteer sample80000 的分层结果：

| 子集 | 路线 | 可解析 | joint-count MAE | J2J | topology F1 |
|---|---|---:|---:|---:|---:|
| train-low (16) | flat UniRig | 11/16 | 5.6364 | 0.056670 | 0.486790 |
| train-low (16) | Puppeteer | 16/16 | 54.2500 | 0.080370 | 0.019744 |
| train-common (16) | flat UniRig | 15/16 | 2.8667 | 0.006266 | 0.951028 |
| train-common (16) | Puppeteer | 16/16 | 5.6250 | 0.013647 | 0.599009 |
| valid-low (20) | flat UniRig | 16/20 | 9.4375 | 0.057437 | 0.452194 |
| valid-low (20) | Puppeteer | 20/20 | 65.2500 | 0.063077 | 0.030999 |

结论：

- flat UniRig 在 common 组已经能生成高质量、可用的骨架，但仍有 10/52 个
  hitmax，尚不能作为无条件可用的最终模型。
- Puppeteer 三个 checkpoint 都能按语法结束，但低关节样本通常生成
  `52/89/96/101` 等大量多余关节；语法成功不等于骨架可用。
- Puppeteer sample80000 与 final 有 50/52 行 generated token 完全相同；
  best_val 只与它们相同 6/52，且总体没有更好。后续以 sample80000 作为同预算
  对照，final 仅视为近等价副本。
- 旧文档中的 `45.12/43.75` 等 joint-count MAE 不对应当前保存的正式结果，
  已被本节 matched evaluation 取代，不得继续引用。

代表样本 montage 位于 `visuals/matched_montage.png`；10 个 flat hitmax 的专门
对照位于 `visuals_flat_hitmax/matched_montage.png`。

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
- 固定同一资产的 pose-A GT 与 GT prefix，只将 condition 换成 pose-B 后，
  train-low/valid-low 前十 joint 的 coordinate NLL 分别恶化约 `0.39~0.49` 和
  `0.48~0.60`，train-common 恶化约 `2.85~2.93`。模型读取了当前 pose，但在
  common rig family 上学到的 pose-to-skeleton 映射明显更强。
- 固定资产、pose 与 GT，只更换 surface/FPS seed，coordinate NLL 平均绝对变化
  只有约 `0.02~0.04`。FPS 随机性不是系统性失败的主因。
- GT-prefix 下 train-low/valid-low 的 joint 0 coordinate accuracy 分别为
  `0.396/0.400`，后续 joint 多数更差。自由生成首次错误落在 joint 0 是序列
  起点效应，不是 rootless joint 0 独有的表示错误。
- 训练集 52-joint 行共有 `5,122` 行，其中同一个完整 parent topology 出现
  `4,930` 次；28-joint 与 34-joint 的主 topology 分别出现 `1,337/786` 次。
- valid-common60 上，目标 topology 训练频率 `>=100` 的样本 coordinate NLL、
  count MAE、J2J、F1 分别为 `0.910/3.29/0.0108/0.672`；训练集中未见 topology
  分别为 `2.502/31.00/0.0433/0.129`。
- 控制 target joint count 后，`log(1 + topology frequency)` 与 coordinate NLL
  和 F1 的 partial Spearman 仍为 `-0.829/+0.741`。因此这不是 joint-count
  分桶能够解释或修复的问题。
- 52-joint valid 样本中 19/20 自由生成 52 joints；所有 heldout-52 中生成
  52、28、34 joints 的结果都落入对应的主训练 topology。当前模型学到的是
  高频 rig-family 模板选择和 pose 调整，而不是长尾 topology 上普适的
  condition-to-skeleton 映射。
- 低 joint 边界的经验停止率只有约 `0.28%~0.74%`。长尾映射欠拟合后，自回归
  rollout 会被强 continuation prior 放大并停在 52/100/101 等长度 hazard。

完整因果证据、数值和文件路径见
`docs/PUPPETEER_TOPOLOGY_LONG_TAIL_DIAGNOSIS_20260718.md`。

## 9. 当前仍未解决的问题

1. 尚未验证在保持架构、数据契约和初始化不变时，直接平衡完整 parent topology
   family 的梯度质量，能否恢复长尾 condition-to-skeleton 映射且不破坏 52-joint
   高频 family。
2. topology-family 频率是已确认的强因果候选和质量预测变量，但仍可能代理某些
   topology 内在难度。下一次受控采样实验负责区分“频率不足”与“表示能力不足”。
3. flat UniRig 的 10 个 hitmax 已定位为 GT-prefix 下多数会 EOS、self-prefix
   下 EOS 被抑制的 exposure mismatch；如何修复而不产生垃圾骨架仍未验证。
4. 尚未证明哪种 decoder/表示能同时保留 flat UniRig 的拓扑先验与显式 parent
   index 的确定连接优势。

## 10. 继续工作的硬规则

- 当前只允许从 Puppeteer `sample80000` 做一次短程拓扑族采样因果探针；它不能
  被称为 baseline，也不能替代同预算干净初始化训练。
- 允许从官方静态 UniRig checkpoint 重训 flat UniRig 对照线；这不依赖旧动态
  Evoweave checkpoint，也不改变 Puppeteer。
- 任何“变好”结论必须来自同一 manifest、同一 pose、同一归一化、同一指标实现
  的 UniRig/Puppeteer 对照，并附自由生成可视化。
- teacher-forcing accuracy、单独 CD 数值或 joint count 改善都不能作为验收。
- 新改动必须先通过 query/target 对齐、prefix logits、梯度和小规模自由生成检查；
  未通过时不得启动正式多卡任务。
- 最终验收必须同时检查当前 pose 对齐、joint count、EOS/hitmax、parent tree、
  J2J/J2B/B2B 和 GT/Prediction/Overlay。

## 11. 下一项唯一允许的执行工作

同预算 flat UniRig 重训、四方 matched evaluation 和逐 token 因果诊断已经完成。
下一项只允许：

1. 以完整 `target_parents` tuple 定义 topology family；
2. 实现 natural row-uniform 与 topology-family-uniform 的 mixture sampler；
3. 使用 `sequence_mean` CE，避免短 skeleton 因 token 少再次被降权；
4. 第一轮保持 termination auxiliary weight 为 `0`，不混入 EOS 新变量；
5. 从 Puppeteer `sample80000` 进行 200~300 step 短程全参数因果探针；
6. 同时评测 heldout-52 与 valid-common60，并按目标 topology 训练频率分层；
7. 只有长尾 coordinate NLL、count、J2J、F1 同时改善且 52-joint 高频 family
   不明显退化，才允许安排同预算、干净初始化的正式训练。

不得重复 joint-count-bin sampler、单独 sequence-mean、termination auxiliary 或
root/joint-0 probe；它们已经回答过不同问题。数据、tokenizer、condition 与
decoder 架构契约保持不变。
