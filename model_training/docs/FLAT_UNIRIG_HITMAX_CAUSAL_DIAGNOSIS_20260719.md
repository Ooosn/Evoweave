# Flat UniRig hitmax 因果诊断

状态：2026-07-19 已完成固定 checkpoint、固定 heldout-52、逐 token
条件干预、前缀修复、重复模式、训练长度分布和采样确定性检查。

## 1. 结论

`hitmax` 不是单独的 EOS 分类错误，也不是“第一个坐标预测错了”这么简单。
完整机制分为两段：

1. 对低频或未见 topology，模型在正确 GT prefix 下的 condition-to-skeleton
   映射已经较弱。自由生成的早期坐标误差会把 prefix 带到训练中没有出现过、
   但 flat tokenizer 语法仍然允许的状态。
2. 进入这种 self prefix 后，decoder 越来越依赖自己已经生成的 token，
   mesh condition 对坐标、branch 和 EOS 的控制力随距离衰减。flat tokenizer
   又允许无限追加坐标三元组，没有 parent index、覆盖状态或重复关节约束，
   最后模型进入常量或周期坐标环，EOS 长期维持低概率，直到 1400-token 上限。

所以：

```text
长尾映射不足
-> 早期 off-manifold self prefix
-> prefix 逐渐压过 mesh condition
-> grammar-valid 的重复坐标吸引子
-> EOS 无法恢复
-> hitmax
```

`hitmax` 是上述链条的末端现象，不是根因。

## 2. 固定对象与定义

本轮只检查：

```text
checkpoint:
/home/wangyy/evorig/outputs/flat_unirig_hgc2h100_matched80k_20260717/
checkpoint_sample_80000.pt

manifest:
/home/wangyy/evorig/outputs/puppeteer_identity1024_preln_hgc2h100_full_20260715/
length_balance_probe/heldout_52.jsonl

saved generation:
/home/wangyy/evorig/outputs/matched_heldout52_20260718/
flat_sample80000.json

seed: 101
generation limit: 1400 tokens
```

术语：

- `GT prefix`：预测下一个 token 时，前面输入 decoder 的 token 全部来自正确
  target。
- `self prefix`：前面输入 decoder 的 token 来自模型自己的 greedy generation。
- `condition swap`：保持同一条 prefix 不变，只把 1024 个 mesh/motion condition
  token 换成另一资产的 condition。
- `zero condition`：保持 prefix 不变，把 condition 置零。
- `first mismatch`：自由生成序列第一次与 target token 不同的位置。它是进入
  错误轨迹的边界，不自动等于根因。
- `hitmax`：在 1400-token 上限前没有生成 EOS。

## 3. 正确前缀下 condition 确实有效

正确 GT prefix 下，正确 condition 明显优于 swapped condition：

| 指标 | correct | swapped |
|---|---:|---:|
| 全部 coordinate NLL | 1.663 | 3.800 |
| joint 0 coordinate NLL | 2.287 | 8.211 |
| EOS NLL | 0.507 | 2.341 |

因此已经排除：

- 输入 mesh 与 target 骨架整体错配；
- decoder 从一开始完全忽略 condition；
- 训练始终使用 reset-pose target。

但是 10 个 hitmax 行即使在 GT prefix 下也弱于成功行：

| 指标 | hitmax 10 行 | success 42 行 |
|---|---:|---:|
| teacher joint-0 coordinate NLL | 2.907 | 2.141 |
| teacher EOS accuracy | 0.600 | 0.905 |

这说明自由生成前，长尾映射和结束决策就已经存在质量差距。

## 4. 同一错误 prefix 上，condition 控制力逐渐消失

对每个 hitmax 样本，固定保存下来的完整 self prefix，只替换 condition。
正确 condition 与 swapped condition 的 top-1 一致率如下：

| 距 first mismatch 的 token 距离 | coordinate 一致率 | action group 一致率 |
|---|---:|---:|
| 0--3 | 0.275 | 1.000 |
| 4--15 | 0.555 | 0.778 |
| 16--63 | 0.706 | 0.844 |
| 64--255 | 0.943 | 0.980 |
| 256--511 | 1.000 | 0.995 |
| 512+ | 0.915 | 0.996 |

`action group` 只区分 coordinate、branch 和 EOS 等粗粒度 token 类别。
0--3 区间 action-group 样本很少，且同为 coordinate 不代表坐标相同，因此
不能把该处的 `1.000` 解读成正确结构。

关键事实是：前 64 个 token 内，换 mesh 会明显改变坐标；进入长 self prefix
后，换成另一资产仍产生几乎相同的坐标和结构决策。此时 decoder 的近期 prefix
已经压过 condition。

在目标 joint count 处，10 个 hitmax 行的 EOS 概率为：

| condition | EOS probability |
|---|---:|
| correct | 0.00534 |
| swapped | 0.03324 |
| zero | 0.03225 |

正确 mesh 反而不能把模型从当前错误 prefix 拉回 EOS。说明问题不只是 condition
强度不足，而是模型从未学过“给定错误但语法合法的 partial skeleton，如何恢复”。

## 5. hitmax 是明确的重复坐标环

| 指标 | hitmax 10 行 | success 42 行 |
|---|---:|---:|
| trigram unique ratio | 0.0466 | 0.8408 |
| 末尾相同 token 连续长度 | 342.1 | 1.0 |
| suffix periodic coverage | 0.2519 | 0 |

10 个 hitmax 中有 7 个最终把同一个 coordinate token 连续重复
`302--780` 次。flat tokenizer 每三个坐标 token 就可以解释为一个新 joint；
它没有以下信息：

- 显式 parent index；
- 已覆盖或已生成 joint 的集合状态；
- duplicate joint 或 zero-length bone 的结构有效性；
- 与 mesh 尚未解释区域相联系的停止条件。

因此“无限重复同一坐标”在 token grammar 中仍然合法。模型不是在生成一套可用
但稍长的骨架，而是在语法允许的无效吸引子里循环。

## 6. 只修第一个错误不能恢复

对 10 个 hitmax 行执行有界自由生成：

| 干预 | EOS rate | 可解析行 J2J | 可解析行 topology F1 |
|---|---:|---:|---:|
| 只替换 first mismatch token | 0.30 | 0.0814 | 0.493 |
| 强制第 1 个 GT joint | 0.50 | 0.0725 | 0.288 |
| 强制前 2 个 GT joints | 0.60 | 0.0535 | 0.608 |
| 强制前 4 个 GT joints | 0.80 | 0.0311 | 0.703 |

干预不是单调的：同一样本可能强制 2 joints 后结束，强制 4 joints 后反而再次
循环；即使生成 EOS，joint count 和 topology 也可能仍然错误。

因此已经排除“step 1 修好，后面自然全部正确”这一解释。强制 prefix 或强制
EOS 只能作为诊断，不能作为模型修复或可用骨架保证。

## 7. 长度监督稀疏会放大循环

按当前 flat tokenizer 对全部 15,920 条 train target 精确计算：

```text
token length = 3 + 3 * joint_count + 4 * nonsequential_branch_count
```

| 分位点 | target token length |
|---|---:|
| min | 15 |
| median | 207 |
| p90 | 264 |
| p95 | 349 |
| p99 | 587.81 |
| max | 1362 |

超过 256 tokens 的训练行只有 `11.26%`，超过 512 的只有 `1.53%`，
超过 1024 的只有 `0.088%`。decoder 最大位置是 3076，所以这不是 position
overflow；真正的问题是错误 rollout 进入了几乎没有 target 监督的长位置区间。

baseline 的 recovery、generated-prefix、oracle-prefix、structure-action 和
condition-control 辅助 loss 全部为 0。训练只有 GT-prefix teacher forcing，
没有任何目标教模型如何从这种长、错误、重复的 self prefix 中退出。

## 8. FPS 不确定性是触发器，不是主因

固定 seed 重建 condition 时，唯一不完全复现的行是：

```text
2363c4b46ee343b7b6b27a225f0decff_seq0.npz
target joints: 4
```

该 mesh 有 `167,703` 个 faces，其中 `9,537` 个 face area `<=1e-8`。
CUDA `torch.multinomial` 在这种极端面积分布上没有得到 bitwise-identical
face samples；约 `51/57,344` 个 sampled face index 不同，随后 FPS 的
`476/1024` 个 anchor index 不同，condition 最大绝对差约 `11.75`。

但是两套 condition 的完整自由生成都：

- 没有 EOS；
- 达到 1400 tokens；
- 生成约 460 joints；
- 进入长常量 token 尾部。

两套生成的前 819 个新 token 完全相同，1400 个位置中有 1388 个相同；差异只在
循环尾部改变了常量 token 的具体切换位置。

采样变化只改变循环的具体轨迹，没有把失败变成功。它说明模型对 surface
sampling 扰动不够鲁棒，但不能解释整个 hitmax 机制。

## 9. 已排除的伪修复

以下做法不能作为结论或最终修复：

- 增大 `max_new_tokens`：只会让循环更长。
- 到上限强制 EOS：输出仍可能是重复坐标垃圾骨架。
- 按目标 joint count 强制 EOS：推理时没有 GT count。
- 只纠正第一个 token：前缀修复实验已证明不稳定且不保证 topology。
- 固定 FPS seed：不能消除已验证的失败。
- 只提高 EOS loss：没有定义错误 partial tree 下应该如何恢复。
- 继续只做 topology 重采样：已有 probe 改善长度但损害 common geometry。

## 10. 新方案必须满足的契约

下一版设计必须同时解决“结构决策依赖 condition”和“错误 partial tree 可恢复”：

1. 保留 flat UniRig 已验证的坐标生成先验，不因消除 hitmax 而牺牲骨架质量。
2. parent 与 stop/continue 必须直接读取 mesh condition 和当前 partial tree，
   不能只由最近的坐标 token 隐式决定。
3. partial tree 状态必须能表达显式 parent、已生成节点和重复/零长度结构，
   使无限追加同一点不再是合法高概率路径。
4. 自前缀训练不能把任意错误第 `k` 个 joint 强行绑定到 GT 第 `k` 个 joint。
   需要先把当前生成的 partial graph 与 GT graph 做语义匹配，再定义剩余结构、
   parent、坐标和 EOS 的监督。
5. 任何新方案在正式训练前必须通过：
   - GT-prefix condition swap；
   - self-prefix condition swap；
   - duplicate/zero-length partial-tree validity；
   - bounded free generation；
   - heldout-52 与 valid-common60 同协议质量；
   - GT/Prediction/Overlay 可视化。

## 11. 完整证据

```text
/home/wangyy/evorig/outputs/matched_heldout52_20260718/diagnostics/
flat_hitmax_causal_complete_seed101_20260719.json
flat_hitmax_causal_complete_seed101_20260719.log
flat_condition_determinism_first7_replay_seed101_20260719.json
flat_condition_determinism_first7_replay_seed101_20260719.log
```

生成诊断的代码：

```text
model_training/analysis/diagnose_flat_autoregressive_failures.py
```
