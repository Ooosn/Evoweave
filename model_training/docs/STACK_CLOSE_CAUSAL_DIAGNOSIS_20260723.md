# Stack-Close 因果诊断（2026-07-23）

## 1. 当前问题

正式 flat UniRig baseline 仍是当前可用基线。旧 stack-close 和 condition-refresh
路线没有被接受为改进模型，因为它们虽然减少了传统 hitmax，却把一部分错误变成了
超长但带 EOS 的树、重复 joint 和较差的坐标。

本轮目标不是继续增加模块，而是回答两个问题：

1. stack 表示本身是否解决了 flat `parent xyz` 表示的问题；
2. stack 新增的失败来自训练配置、CHILD/CLOSE 决策，还是 condition 在自由 rollout
   中失效。

在这些问题得到固定样本、固定 pose、固定 surface/FPS 的证据之前，不提交新的正式
双卡训练。

## 2. 固定基线

共同验证集为 rootless-v3 `valid_manifest.jsonl` 的 857 行。

| 路线 | 合法生成 | 主要失败 | 全部行 topology F1 | J2J | count MAE |
| --- | ---: | --- | ---: | ---: | ---: |
| flat UniRig, sample 80000 | 780/857 | 77 true hitmax | 0.7428 | 0.0172 | 6.997 |
| 旧 stack-close, sample 80000 | 838/857 | 19 invalid，另有 73 个极端过生成 | 0.6913 | 0.0341 | 25.38 |

旧 stack 不是“已经修好 hitmax”。它主要把无法结束改成了很晚才结束：73 行生成约
267--272 个 joint 或出现同类极端过生成，因此必须把合法 EOS、关节数量、F1、J2J
和可视化一起验收。

## 3. 已证实事实

### 3.1 Stack 表示有真实收益，但旧训练又制造了新失败

用 `invalid OR pred_joint_count - target_joint_count > 50` 定义旧 stack 的极端失败，
flat 与 stack 的 857 行交叉表为：

| flat | stack | 行数 |
| --- | --- | ---: |
| 成功 | 成功 | 729 |
| 成功 | 失败 | 51 |
| 失败 | 成功 | 36 |
| 失败 | 失败 | 41 |

因此 stack 救回了 77 个 flat 失败中的 36 个，证明显式 DFS 栈、取消 `parent xyz`
反查不是无效改动。但旧 stack 同时新制造了 51 个失败，所以旧训练方式不能保留。

### 3.2 失败不是单纯的长序列问题

旧 stack 极端失败行的目标 joint 数中位数为 33.5，成功行为 52；最大深度中位数分别
为 8 和 10。目标 joint 数 4--27 的组失败率最高（约 26.4%）。因此不能用“树太长”
解释主要失败。

### 3.3 随机 sibling order 是有标签噪声的目标

训练集中 15,756/15,903 行包含分叉；同一个正确 prefix 会因 sibling 随机排列而对应
不同的下一组坐标。共有 69,422 个 branching parent、256,476 个 joint 位于这种歧义
集合中，约占全部 joint 的 34.53%。估算的不可约 CE 约为 0.07325/token。

控制 target joint count 后，sibling entropy 对 count error 和 F1 仍只有弱相关。因此
canonical sibling order 是必要清理，但不能单独解释或修好全部失败。

### 3.4 旧正式 stack 还意外改变了 conditioner

flat baseline 明确使用：

```text
use_motion_features = false
use_time_embedding = false
```

旧 stack 使用了 `true/true`，额外增加 1,115,136 个可训练参数。13 维 raw
position/rest/delta/velocity/norm 经带 LayerNorm 的 MLP 以完整幅度加到 mesh token；
time embedding 又编码了 FPS + random evidence 的采样槽位，而这些槽位不是可靠的时间
顺序。这不是 tokenizer 对照中应该出现的变量。

### 3.5 不是简单的学习率过高或没有收敛

旧 stack 在约 5k/20k/50k/80k 样本曝光处的 AR 学习率约为：

```text
7.28e-5 / 9.32e-5 / 3.84e-5 / 2.00e-6
```

同期 teacher-forcing CE 从约 1.57 降到 1.13，但困难组重复 joint 和过生成继续增加。
这是 teacher-forcing 目标持续改善、自由 rollout 持续恶化的分布差异，不是单纯训练
不足。

### 3.6 弱 condition 是风险因素，但不是唯一原因

控制数据源和 joint-count 区间后，旧 stack 失败与以下量有弱到中等关联：

- skin 顶点到 bone head 的中位距离更大；
- frame bbox 变化更小；
- joint/vertex motion 更弱。

这些相关性不足以当作新的数据 reject gate，也不能证明数据错误。它们支持的结论仅是：
当 motion 对骨架的约束较弱时，自回归 decoder 更容易依赖 prefix 并错误续写。

### 3.7 Condition refresh 不是已验证修复

旧 condition-refresh 的 zero gate 到 72k 样本曝光时最大绝对值仅约 0.00528；它在
857 行上修好部分旧 stack 失败的同时又制造新失败，总体没有稳定优势。因此它不进入
当前候选正式路线。

## 4. 当前受控实验

固定诊断根目录：

```text
/ssdwork/liuhaohan/evoweave/outputs/stack_close_causal_diagnosis_20260723
```

固定输入契约：

- query/evidence seed: `20260722`；
- surface/FPS base seed: `20260722`；
- surface seed 由 manifest index 唯一派生，评测顺序不改变输入；
- 32 行诊断集：16 flat-hitmax、8 flat-good/stack-bad、8 双方正常控制；
- 三组内部按 target joint count 分层抽取。

### 4.1 Clean/canonical 5k 对照

只相对旧 stack 改两类已知混杂变量：

```text
random sibling -> canonical sibling
motion/time extras true/true -> false/false
prefix perturbation -> 0
```

初始化、数据、有效 batch、样本曝光数和 OneCycle 调度保持一致。该试跑用于判断旧
stack 的训练配置是否已在早期制造优化负担，不作为最终性能结论。

### 4.2 冻结旧 stack 的二分类 action 诊断

固定同一个旧 stack-80k 基础模型，只训练 `CHILD` 对 `CLOSE` 的 action head：

1. query-only：只读 decoder 当前 hidden state；
2. condition-aware：除 hidden state 外，直接 cross-attend 原始 1024 condition token。

训练集 action 标签基本平衡：train CLOSE 占 50.54%，valid 占 50.55%。两个 head 都在
未参与训练的固定 valid 32 行上自由生成。

判别规则：

- 两者都改善：主要问题是 257 类 token softmax 混合了拓扑动作与坐标；
- 只有 condition-aware 改善：自由 rollout 中 decoder hidden 的 condition 信息不足；
- teacher-forcing action 高、自由生成仍失败：主要问题是 self-prefix hidden 分布漂移；
- 控制组提前截断或 J2J/F1 恶化：判定 action 修复失败，不能以“无 hitmax”通过。

## 5. 正式双卡提交门槛

只有同时满足以下条件，才使用一次 `2 x A100-80GB` 正式训练机会：

1. 固定 flat-hitmax 组的合法率、count error 和 topology F1 明显改善；
2. flat-good/stack-bad 组不是只从过生成变成欠生成；
3. 双方正常控制组的 F1/J2J 不出现实质退化；
4. 可视化骨架覆盖 mesh，不能只看 EOS 或 aggregate F1；
5. 训练与生成调用同一个 action 定义，optimizer 参数覆盖审计通过；
6. 正式路线显式关闭随机 sibling、raw motion features、time embedding、旧
   condition refresh 和所有 fallback。

当前状态：正式双卡任务尚未提交。
