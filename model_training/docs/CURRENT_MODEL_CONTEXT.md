# 当前模型任务（唯一入口）

更新时间：2026-07-24

状态 ID：`model-local-motion-evidence-20260724-001`

本文是模型模块唯一的当前状态入口。旧的 Puppeteer、stack-close、
condition-refresh、anchor residual、centered-motion、query-rigid 和 HGC
重训记录均不是当前任务，不能据此启动实验。

## 1. 当前目标

当前目标不是继续修补 EOS，也不是继续设计全局 motion residual。目标是检验并实现：

```text
query pose 的静态 mesh token
+
只描述局部表面相对运动的 articulation evidence
-> 当前 pose 的完整 rootless skeleton
```

核心问题是：现有 motion encoder 输出的 1024 个 query-aligned condition token
混合了静态外形、动作幅度、局部相对运动和网络产生的特征偏移。最终骨架损失没有要求
motion encoder 明确学习“哪些表面区域由同一骨段控制”，所以模型会优先使用静态外形和
高频骨架模板，只不稳定地使用 motion。

当前数据在训练输入中已经去掉全局 RT；全局平移或旋转不是本轮数据问题，也不能作为
本轮方法的主要动机。

## 2. 唯一数据来源

只使用西湖 rootless-v3 最终 manifest：

```text
train:
/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests/train_manifest.jsonl

valid:
/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests/valid_manifest.jsonl
```

固定数量：

```text
train = 15903
valid = 857
```

必须读取 manifest 中的 `path`，不得重新扫描 NPZ 目录。训练 target 没有 synthetic root，
没有 tail token；`joint 0` 是 rootless 后唯一真实树根。query mesh 和 target skeleton 必须
来自同一个随机 query pose，并使用同一个 query-bbox 坐标系。

## 3. 当前对照模型

当前唯一对照是西湖训练的 flat UniRig baseline：

```text
/ssdwork/liuhaohan/evorig/evoweave_repo/outputs/dynamic_rig_runs/
rootless_flat_unirig_motion_fullft_20260707_hxr4gpu/
checkpoint_sample_80000.pt
```

该 checkpoint 只用于开发阶段的同服务器对照和接入预检。不得用 HGC checkpoint 替换，
也不得从历史 Puppeteer、stack-close 或 residual checkpoint 初始化当前方法。

正式新模型若获得训练许可，应与该 baseline 对齐数据、官方 UniRig 静态初始化、batch、
学习率计划和样本曝光，并全参数训练；唯一结构变量应是经过预检的 motion-evidence 路径。

## 4. 当前模型设计

静态路径保持现有 query pose 的 1024 个 mesh token：

```text
Q_i = 第 i 个 query-pose surface token
```

运动路径不读取 `Q_i`，也不读取绝对位置作为可学习内容。对 query surface 上的局部邻接点
`i, j`，构造跨帧相对距离变化：

```text
r(i,j,t) = ||x_i(t) - x_j(t)|| / (||x_i(q) - x_j(q)|| + eps) - 1
```

这些局部关系经过时间汇总或轻量编码后形成：

```text
E_i = 第 i 个表面区域的局部 articulation evidence
```

`E_i`必须满足：

- 重复 query pose 的零 motion 输入产生零证据；
- 不包含足以单独复制 query mesh 的绝对静态信息；
- 同一骨段内部关系通常稳定；
- 跨骨段或蒙皮边界的关系随有效动作变化；
- 低 motion 表示证据不足，不表示不存在关节。

训练数据中的 skin weight 只用于辅助监督。局部点对的 soft same-segment label 定义为：

```text
s(i,j) = sum_k min(w_i[k], w_j[k])
boundary(i,j) = 1 - s(i,j)
```

推理阶段不读取 skin weight。

静态 token 和 motion evidence 必须作为两套独立 memory 被 decoder 查询，不能预先相加成
anchor residual：

```text
当前 skeleton prefix h_k
  -> cross-attention(query static tokens Q)
  -> cross-attention(local motion evidence E)
```

第一个真实 joint 主要由 query mesh 定位；后续 child/branch 再使用局部 motion evidence
消除结构和关节位置歧义。

详细设计与验收定义见：
`model_training/docs/EVIDENCE_AWARE_MOTION_CONDITIONING.md`。

## 5. 当前执行顺序

当前只允许按以下顺序工作：

1. 在西湖正式 NPZ 上构造局部相对运动特征和 skin-relation label。
2. 以 asset-disjoint train/valid 划分验证 motion-only evidence 是否有未见样本增量。
3. 对比正确 correspondence、重复 query pose 和打乱 anchor correspondence；不能把时间顺序
   打乱当作负例，因为目标证据可以是时间顺序不变的。
4. 只有 evidence 本身成立，才实现独立 motion-evidence branch。
5. 接入后检查 query/target 对齐、参数梯度、零 motion 行为、正确/错误 motion 干预和小规模
   自由生成可视化。
6. 全部通过后，才允许提交唯一一次双 A100 正式训练任务。

第一阶段不需要加载任何 baseline checkpoint。checkpoint 只在第四、第五步接入生成模型时
使用。

## 6. 明确排除的上下文

以下内容不得作为当前下一步：

- HGC flat-UniRig 或 Puppeteer checkpoint；
- centered-motion 公式或其 selected18 评测；
- anchor/static residual 微调；
- query-rigid 推理修复；
- stack-close、sibling perturb 或 condition-refresh 继续训练；
- 强制 EOS、长度裁剪、oracle-prefix 或显式 parent-index 修补；
- 把全局 RT 当作当前数据错误。

这些历史路线只能解释为什么当前任务转向局部 articulation evidence。

## 7. 资源边界

当前西湖开发环境有一张 A100 80GB。正式双 A100 训练机会剩余一次，但在 motion evidence
预检和接入预检完成前，`train` 与 `submit` 均被状态锁阻止。不得使用
`huangxiangru` 资源组提交新任务。
