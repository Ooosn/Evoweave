# Local Motion Evidence for Skeleton Generation

状态：当前设计，尚未进入正式训练。

## 1. 假设

现有模型已经读取 motion，但没有稳定地把它变成骨段关系。当前假设是：

```text
局部表面相对运动
-> 同骨段关系、蒙皮边界和关节证据
-> 更可靠的 skeleton condition
```

该假设必须先独立于自回归模型得到验证。不能用一次完整训练来代替特征有效性检查。

## 2. 输入与表示

每个样本沿用训练 loader 的 query pose、帧选择和 1024 个对应 surface anchors。静态路径
继续产生 query tokens `Q`。

运动路径只处理局部关系变化。对局部邻接点 `i,j`：

```text
r(i,j,t) = ||x_i(t)-x_j(t)|| / (||x_i(q)-x_j(q)|| + eps) - 1
```

邻接关系必须来自 query surface 的有效局部结构，并保持 mesh component 边界；不得用会把
空间上接近但表面不连通区域随意连接起来的隐式 fallback。

跨帧描述可以保留完整短序列，也可以使用经过验证的 robust statistics；选择必须由
held-out evidence 实验决定，不能凭直觉在正式训练时临时更改。

## 3. 训练监督

对每个局部点对，使用 rootless NPZ 中已有的 skin weights：

```text
same_segment(i,j) = sum_k min(w_i[k], w_j[k])
boundary(i,j) = 1 - same_segment(i,j)
```

这是 soft、joint-order-independent 的监督。低 motion 样本不能被当作 boundary=0 的负例；
其辅助损失需要按可观测运动证据降权或标记为 unknown。

skin weights 只在训练阶段生成辅助 label，推理阶段不存在该输入。

## 4. 第一阶段验收

第一阶段只验证 evidence，不加载 skeleton checkpoint。必须报告：

- asset-disjoint valid AUROC、AUPRC 和校准结果；
- 按 motion amount 分层的结果；
- 正确 correspondence；
- 重复 query pose 的零 motion；
- 保留运动幅度但打乱 anchor correspondence 的错误 motion；
- feature extraction 的每样本耗时和内存；
- 代表性 mesh 上的 evidence/skin-boundary 可视化。

验收逻辑不是要求零 motion 一定获得最低分类分数，而是要求模型能够区分“可观察证据”和
“未知”，且正确 correspondence 在有充分运动的样本上明显优于错误 correspondence。

## 5. 生成模型接入

第一阶段成立后，才增加独立 evidence encoder：

```text
Q: query-pose static memory
E: local-motion evidence memory

h_k -> cross-attention(Q)
h_k -> cross-attention(E)
```

`E`不能在进入 decoder 前与 `Q`相加，也不能由同时读取 `Q`和 motion 的 MLP 生成。这样
才能排除旧 anchor residual 的静态捷径。

主 skeleton token/loss 保持 baseline 语义。新增 skin-relation auxiliary loss 只约束
evidence encoder，不改变推理输入和 GT skeleton contract。

## 6. 正式训练门槛

提交双 A100 前必须同时满足：

1. evidence 在 held-out assets 上有稳定增量；
2. 零 motion 与错误 correspondence 控制符合定义；
3. evidence encoder 和 decoder evidence-attention 均有有限非零梯度；
4. query/target pose、FPS、坐标归一化完全一致；
5. 正确 motion 对 child/later-joint 的影响优于错误 motion，且不破坏 root；
6. 小规模自由生成在失败样本上改善，同时不明显伤害匹配正常样本；
7. GT、Prediction、Overlay 可视化通过人工检查。

在这些条件满足前，不允许把 centered-motion、anchor residual 或任何历史 checkpoint 当作
该方法的替代品。

## 7. Held-out evidence result (2026-07-24)

The evidence-only gate passed on the Westlake rootless-v3 manifests without
loading a skeleton checkpoint. The diagnostic used 4,096 train assets and all
857 validation assets, with 1,024 uniformly sampled real mesh edges per asset.
The complete manifests contain 15,903 unique train asset ids and 857 unique
validation asset ids with zero overlap.

At soft-boundary threshold 0.25:

```text
correct correspondence global AUROC        0.7499
correct correspondence per-asset median    0.8162
within-asset corrupted per-asset median    0.5024
repeated-query zero-motion AUROC            0.5000
raw edge-motion RMS AUROC                   0.7478
11-feature diagnostic MLP AUROC             0.7499
```

Absolute motion stratification confirmed the uncertainty contract. For the 89
validation assets whose per-example q90 edge-motion RMS was below `1e-3`, AUROC
fell to 0.5488. The other motion ranges remained between 0.7142 and 0.7486.
Low motion is therefore unknown evidence and must attenuate the evidence path;
it must not be trained as a negative structural label.

The diagnostic MLP improved AUROC over raw RMS by only 0.0021. It is not part
of the proposed model. The validated primitive is the topology-constrained
relative deformation itself. The next model should encode simple local
statistics and learn how the skeleton decoder uses them, rather than presenting
the diagnostic classifier as a contribution.

The evidence memory also needs an explicit addressing contract. Motion-only
values do not contain enough static information to identify a location by
themselves. Decoder evidence attention will therefore use aligned static query
tokens as keys and motion-only evidence as values:

```text
query = current skeleton-prefix state
key   = aligned query-pose static token Q_i
value = topology-local relative-motion evidence E_i
```

This allows spatial lookup without adding static geometry into the evidence
value. It remains different from the rejected static-plus-motion residual.

Canonical output:

```text
/ssdwork/liuhaohan/evorig/outputs/
local_motion_evidence_preflight_20260724_train4096_valid857
```
