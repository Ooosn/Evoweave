# 骨架生成结构预检与下一轮训练边界

状态：2026-07-19，单张 H100 的 conditioner、decoder rollout 与 tree 诊断，以及
flat tokenizer 的全量 train/valid round-trip 审计均已完成。本轮没有训练新模型。

## 1. 这轮要回答什么

这轮不训练新模型，只回答下一次大规模训练前必须弄清的三个问题：

1. 当前 1024 个 mesh/motion condition token 是否包含骨架结构信息。
2. condition 进入 flat UniRig decoder 后，是从一开始就没有作用，还是在
   self-prefix 变长后失去控制力。
3. 真实数据的 tree、connector、高分叉和重合 joint，允许我们采用什么结构表示。

术语定义：

- `condition token`：动态 mesh conditioner 输出的 1024 个向量，每个向量维度与
  UniRig decoder hidden size 相同。
- `GT prefix`：预测下一个 token 时，已经输入的骨架 token 全部来自正确 target。
- `self prefix`：已经输入的骨架 token 来自模型自己的 greedy generation。
- `condition swap`：保持同一条骨架 prefix 不变，只把 1024 个 condition token
  换成另一资产的 condition。
- `JS divergence`：两个 next-token 概率分布的 Jensen-Shannon divergence。
  越接近 0，表示换 mesh 后输出分布几乎不变。
- `connector`：自身没有皮肤权重，但为了连接有权重后代而必须保留的真实 tree
  节点。
- `persistent zero edge`：parent 与 child 在所有保存帧中都保持同一坐标的边。

## 2. 固定对象

模型：

```text
/home/wangyy/evorig/outputs/flat_unirig_hgc2h100_matched80k_20260717/
checkpoint_sample_80000.pt
```

因果生成集合：

```text
/home/wangyy/evorig/outputs/puppeteer_identity1024_preln_hgc2h100_full_20260715/
length_balance_probe/heldout_52.jsonl
```

结构审计使用与该 checkpoint 对齐的 HGC rebuild：

```text
/home/wangyy/evorig/evoweave_rebuild_rootless_v3_hgc_20260714/
```

该 HGC rebuild 实际是 `train=15920`、`valid=856`、总计 `16776`。工作区当前
source-of-truth 记录的是西湖正式数据 `train=15903`、`valid=857`、总计 `16760`。
本轮为了不改变 checkpoint 的训练分布，诊断使用 HGC 的 `16776`；下一次正式训练
前必须同步并冻结唯一 manifest，不能继续让两个计数并存。

## 3. 当前 decoder 的真实数据流

当前 flat baseline 的数据流是：

```text
动态 mesh
-> 1024 condition tokens
-> [condition 1024, BOS, class, 已生成骨架 token]
-> 24 层 decoder-only causal Transformer
-> 下一个 flat skeleton token
```

condition 不是单独的 cross-attention memory。它只在序列开头出现一次，后续每一层
都由同一个 causal self-attention 同时读取 condition 和不断增长的骨架 prefix。

训练目标只有正确 GT prefix 下的 next-token cross entropy。baseline 没有监督模型
在语法合法但几何错误的 self prefix 中如何恢复，也没有显式 partial-tree state
告诉 decoder 哪些 joint 已经生成、当前 parent 是谁、哪些 mesh 区域尚未解释。

## 4. 1024 condition 是否包含结构

冻结整个 checkpoint，只训练诊断用的 Ridge 线性读出器。读出器不参与 Evoweave
训练。训练 224 条、验证 111 条，并按 joint count 分成 7 个区间均匀抽样。

| 输入特征 | joint count R2 | joint count Spearman | joint count MAE | median baseline MAE |
|---|---:|---:|---:|---:|
| mesh/motion 粗统计 | 0.217 | 0.465 | 30.392 | 34.207 |
| condition token 均值 | 0.404 | 0.722 | 24.685 | 34.207 |
| condition 均值和标准差 | 0.372 | 0.692 | 24.470 | 34.207 |
| decoder 起始 class hidden | 0.299 | 0.642 | 27.370 | 34.207 |

结论：

1. conditioner 不是完全没有结构信息。它对 joint 数量、leaf 数量和 branch 数量的
   线性可读性明显高于 mesh/motion 粗统计。
2. 该信息只是粗粒度。`root_child_count` 的线性读出接近无效，condition 最近邻的
   exact topology 命中率只有 `12.6%` 到 `13.5%`。
3. condition 经过 `[BOS, class]` 的第一次 decoder 汇聚后，joint-count `R2`
   从 `0.404` 降到 `0.299`。decoder 没有把已有结构信号变得更清楚。
4. 这项 probe 只读取全局均值和标准差，不能证明 1024 个有序 token 中不存在更细的
   局部结构；它能证明的是：粗结构信号存在，但当前 decoder 没有可靠利用它。

## 5. condition 在 self prefix 中如何失效

对 heldout-52 中全部 10 个 hitmax 样本和 10 个按 joint count 匹配的成功样本，
固定模型保存的 self prefix，只替换 condition。两套原始 condition 的平均
relative L2 是 `1.371`，说明 swap 的两套条件本身差异充分。

### 5.1 最终 hidden state

| 距 first mismatch 的 token 距离 | hitmax relative L2 | success relative L2 |
|---|---:|---:|
| 0-3 | 0.717 | 0.758 |
| 4-15 | 0.568 | 0.640 |
| 16-63 | 0.462 | 0.586 |
| 64-255 | 0.359 | 0.562，仅 2 条 success 覆盖 |
| 256-511 | 0.310 | 无 success 覆盖 |
| 512+ | 0.256 | 无 success 覆盖 |

### 5.2 next-token 概率分布

| 距 first mismatch 的 token 距离 | hitmax JS divergence | hitmax top-1 一致率 |
|---|---:|---:|
| 0-3 | 0.408 | 0.275 |
| 4-15 | 0.245 | 0.533 |
| 16-63 | 0.138 | 0.710 |
| 64-255 | 0.0305 | 0.944 |
| 256-511 | 0.0097 | 0.998 |
| 512+ | 0.0232 | 0.912 |

换 mesh 在错误初期会明显改变坐标分布；进入长 self prefix 后，换成另一资产仍得到
几乎相同的 token 决策。这是 prefix 压过 condition 的直接因果证据，不是通过生成
结果反推。

### 5.3 发生在哪些层

hitmax 样本在 0-3 token 区间的 condition relative L2 从第 1 层 `0.362`
逐步升到第 14 层约 `1.027`，随后被上层压到最终 `0.717`。在 512+ 区间，第 1 层
只有 `0.153`，第 14 层约 `0.333`，最终为 `0.256`。

因此有两个同时存在的机制：

1. 距离机制：从第 1 层开始，长 prefix 位置获得的 condition 差异就显著更小。
2. 上层机制：中层已经形成的 condition 差异在 decoder 最后约 10 层中再次衰减，
   最终 token head 更偏向当前 prefix 的局部模板。

这也解释了为什么只加 EOS 权重不够。错误 prefix 下，不仅 EOS 低，整个 coordinate
和 action 分布都逐渐不再受当前 mesh 控制。

### 5.4 GT prefix 与 self prefix 的同位置对照

上面的衰减可能有两种解释：

1. token 位置变长后，decoder 天然无法读取序列开头的 condition。
2. 不是长度本身，而是错误 self prefix 逐渐压过 condition。

为区分两者，在同一批样本、同一 first-mismatch 参考位置上，把 prefix 替换成 GT，
再执行相同 condition swap。hitmax 组结果如下：

| 距 first mismatch 的 token 距离 | self-prefix JS | GT-prefix JS |
|---|---:|---:|
| 0-3 | 0.407 | 0.411 |
| 4-15 | 0.246 | 0.314 |
| 16-63 | 0.138 | 0.206 |

0-3 token 时两者几乎相同；到 4-63 token，GT prefix 下换 mesh 仍明显改变
next-token 分布，而 self prefix 下的影响更快衰减。这个对照覆盖同一批 10 个 hitmax
样本。64+ 区间只有 1 条 GT target 足够长，不能把该区间作为总体统计；这条长样本
仍呈现同一方向，但只作为个例。

因此结论不是“decoder 到固定长度后必然看不到 condition”，而是：

1. GT manifold 上，模型能够持续使用 condition。
2. 一旦进入自己的错误 partial skeleton，prefix 内容会把模型推入另一种状态；
   condition 影响被逐层压小，最后几乎不能改变 action 或坐标决策。
3. 训练缺口确实位于“错误但语法合法的 partial skeleton”分布，而不只是 EOS 类别
   不平衡。

### 5.5 hitmax 不是目标骨架太长

10 个 hitmax 中，9 个 target 只有 4 到 9 个 joint，只有 1 个 target 有 73 个
joint；但自由生成的 joint 平均数为 `371.7`。当生成 joint 数第一次达到正确 target
数量时：

- 5 条继续选择 branch；
- 5 条继续选择 coordinate；
- 10 条都没有选择 EOS；
- 平均 EOS 概率只有 `0.0053`，而 42 条成功样本在相同语义位置的平均值为
  `0.5717`。

GT prefix 下，正确 condition 相比 swapped condition 的统计为：

| 目标 | swapped NLL - correct NLL |
|---|---:|
| 第一个 joint 的 xyz | +4.891，hitmax 组 |
| 结构 action 决策 | +0.377，hitmax 组 |
| EOS | +1.891，hitmax 组 |

正值表示正确 mesh condition 给正确 target 更高概率。因此模型并非完全没有学会
mesh 到骨架的对应关系；它在 GT prefix 上明确偏好正确 condition。

10 个 hitmax 还可进一步拆成两类：

- 4 条在 GT prefix 下连 EOS 都不是 top-1，说明其终止判断本身尚未拟合好。
- 6 条在 GT prefix 下 EOS 是 top-1，但自由生成仍 hitmax，说明正确终止能力存在，
  只是生成轨迹没有到达训练过的 prefix 状态。

已有前缀干预与这个结论一致：强制前 4 个 GT joint 后，10 条 hitmax 的 EOS 成功率
从 0 提升到 `0.8`，可 detokenize 行的 topology F1 为 `0.703`。这不是最终修复，
但证明当前模型存在可用的正确轨迹 basin，也证明问题核心是轨迹稳定性，而不是
checkpoint 完全没学到骨架。

## 6. tree 与 tokenizer 的数据约束

全量 `16776` 条 HGC checkpoint-matched 数据：

| 指标 | 结果 |
|---|---:|
| joint 总数 | 783313 |
| edge 总数 | 766537 |
| ordered topology 数 | 3219 |
| singleton topology 数 | 2402 |
| 最大 topology 频次 | 5183 |
| 每样本 max children p50 / p95 / p99 / max | 5 / 9 / 25 / 124 |
| 含 max children > 8 的样本 | 5.70% |
| connector 数量 | 9062，1.16% joints |
| 含 connector 的样本 | 12.16% |
| 有权重但无 hard-argmax vertex 的 joint | 28964，3.74% skinned joints |
| 含上述 joint 的样本 | 44.79% |
| weighted motion < 1e-2 bbox 的 skinned joint | 6.14% |

所有 9062 个 connector 都没有皮肤权重，但全部连接到至少一个有皮肤权重的后代，
所以不能删除。它们证明局部 skin/motion evidence 不是每个 joint 都具备，模型仍需
结构先验。

高分叉也不是只来自 dummy root。`child_count > 8` 的 1213 个节点中，1083 个节点
自身有皮肤权重；极端样本包含一个有权重 parent 和 124 个有权重 child。因此固定
`K=2/4/8` child slots 都会截断合法数据。可行表示必须允许回到同一 parent 后继续
生成任意数量的 child，再显式关闭该 parent。

frame 0 有 436 条精确零长度 edge：

- 356 条在所有保存帧中都保持重合；
- 98 条 persistent edge 的 parent 与 child 都有皮肤权重；
- 抽查包含同一旋转中心的 hair、shoulder、hand deform bones。它们坐标相同，
  但皮肤权重支持集不同；
- 另有电池辐条类节点在 frame 0 重合，但后续 pose 会分离。

因此不能把 duplicate coordinate 或 zero-length edge 一律判非法。真正缺失的是节点
身份：flat tokenizer 用 parent 坐标反推 parent 时，重合或量化碰撞会让 parent
身份不唯一。表示层必须无歧义地编码 parent 身份；显式 parent index 是一种方法，
DFS 栈转移是另一种方法。不能继续用 parent xyz 反推。

flat tokenizer round-trip 全量统计：

```text
train:
  rows                         15920
  detokenize_ok                15920
  duplicate_quantized_rows      2318  (14.56%)
  quant_parent_child_same_rows  1572  ( 9.87%)
  parent_mismatch_rows           989  ( 6.21%)
  parent_mismatch_edges         4696  (train 全部 728028 edges 的 0.645%)
  worst_parent_mismatch_count    103
  identity edge F1 mean/min       0.99527 / 0.21569
valid:
  rows                           856
  detokenize_ok                  856
  duplicate_quantized_rows       120  (14.02%)
  quant_parent_child_same_rows    83  ( 9.70%)
  parent_mismatch_rows            56  ( 6.54%)
  parent_mismatch_edges          259  (valid 全部 38509 edges 的 0.67%)
  worst_parent_mismatch_count     24
  identity edge F1 mean/min       0.99546 / 0.75362
```

train 的 989 条、valid 的 56 条 parent mismatch 都与
`branch_parent_bad_tie` 完全一致。原因是 branch token 只保存量化后的 parent xyz；
当多个先前节点落入相同量化坐标时，detokenizer 固定选择最后一个节点，原 parent
index 已无法从 token stream 恢复。全部 `16776` 条都能成功 detokenize，所以只
检查“能否读取骨架”会漏掉该错误。

这说明当前训练 target 本身包含表示层噪声：NPZ 的 `target_parents` 正确，但 flat
token target 经过标准 detokenizer 后，在 6.21% 的 train 和 6.54% 的 valid 样本
上不再是同一棵树。平均 identity F1 很高也不能掩盖最坏样本，因为错误集中在有
大量重合/近重合节点的少数树上：错误样本 joint count 中位数为 64，全集中位数为
52；train 最坏样本的 identity edge F1 只有 `0.216`。train 的 989 条错误中只有
197 条含连续空间精确零长度 edge，另外 792 条没有；valid 的 56 条中也只有 5 条
含精确零长度 edge。主体是 256-bin 量化碰撞，所以禁止原始重复坐标不能解决。

同一 valid manifest 又用确定性 query pose 重跑一次，结果仍为 56 条错误：

- 两次错误集合各 56 条，交集 54 条，并集 58 条，Jaccard 为 `0.931`；
- 错误边总数为随机 pose `259`、确定性 pose `251`；
- 逐样本 parent mismatch 数 Spearman 为 `0.965`；
- 849/856 条样本的 mismatch 数完全相同。

因此 pose 会让极少数量化边界样本进出碰撞集合，但主体不是随机 pose 造成的。

对 heldout-52 的完全相同 query pose 做逐行对齐：

| 分组 | 行数 | parent mismatch 行 | duplicate quantized 行 |
|---|---:|---:|---:|
| hitmax | 10 | 0 | 4 |
| success | 42 | 2 | 6 |

因此 parent-xyz 不可逆是必须修复的独立表示缺陷，但不是当前 10 个 hitmax 的直接
原因。hitmax 的直接因果证据仍是第 5 节的 off-manifold prefix 与 condition
失效。两个问题必须同时解决，但不能混成同一个结论。

### 6.1 DFS stack-close 候选表示

对同一批 `16776` 棵最终 rootless tree 做了完整验证。候选表示不再输出
`BRANCH + parent_xyz`，而是：

```text
BOS, class
root_xyz
child_xyz              # 当前 stack top 是 parent，随后 child 入栈
CLOSE_NODE             # 当前节点完成，弹栈
...
CLOSE_NODE             # root 完成
EOS
```

验证结果：

| 指标 | flat UniRig | stack-close |
|---|---:|---:|
| 可精确表示样本 | 16776 | 16776 |
| 不满足 DFS 顺序样本 | - | 0 |
| 全量 token 总数 | 3188167 | 3183580 |
| token 数 p50 | 207 | 211 |
| token 数 p95 | 348 | 319 |
| token 数 p99 | 583.25 | 523 |
| 最大 token 数 | 1362 | 1003 |

全量 target 顺序 `100%` 满足 DFS stack contract。stack-close 总 token 数比 flat
还少 `4587`，不是以显著增长序列换取结构正确性。其原因是：

- flat 对每次非连续 parent 跳转使用 1 个 `BRANCH` 和 3 个 parent-xyz token；
- 全量共有 `196975` 次这类跳转；
- stack-close 删除这些 `787900` 个 token，增加每个真实 joint 一个
  `CLOSE_NODE`，共 `783313` 个。

因此每个 joint 都得到一次局部完成监督，而不再只依赖每棵树唯一的 EOS。
高分叉通过重复生成 child 处理，没有固定 child slot；重合 joint 由栈中的节点
身份区分；connector 与普通真实节点完全一致。最大 target 长度从 1362 降到 1003，
加上 1024 condition 后最大总长度为 2027，在当前不添加额外 prompt token 的路径上
也能落入 2048 position budget。

flat 中 `BRANCH` 只占全量 target token 的 `6.18%`，每棵树的 EOS 合计只占
`0.53%`；stack-close 中 `CLOSE_NODE` 占 `24.60%`。这不是简单提高 EOS loss，
而是把“该局部子树是否已经完成”改成每个节点都有监督的结构决策。

## 7. 下一版不应该做什么

以下方案已经与证据冲突：

1. 固定 FPS。它只改变少数失败轨迹，不消除 hitmax 机制。
2. 到长度上限强制 EOS。它只把循环垃圾截短。
3. 只增加 EOS loss。错误 prefix 下整个输出分布都失去 condition 控制。
4. 禁止重复坐标或零长度边。数据中存在有皮肤权重的合法重合节点。
5. 固定每个 parent 的 child slot 数。真实最大 child count 为 124。
6. 把 self-prefix 第 `k` 个预测 joint 直接绑定 GT 第 `k` 个 joint。发生分叉或顺序
   漂移后，这个 ordinal 对齐没有图语义。
7. 重新随机初始化一套独立 decoder。当前 flat UniRig 已有可用的坐标先验，不应
   再重复 Puppeteer 路线的冷启动问题。

## 8. 建议的下一版结构

下一版应是保留 UniRig 坐标生成能力的 `stack-close UniRig`，而不是另一套随机
skeleton decoder。

### 8.1 表示与状态

生成状态显式保存 DFS 节点栈。坐标 triple 表示“在当前 stack top 下新增 child 并
将 child 入栈”；`CLOSE_NODE` 表示当前节点的子树已经完成并弹栈；root 弹栈后才允许
EOS。

这满足：

1. parent 由 stack top 唯一确定，不再用 parent xyz 反推。
2. 同一 parent 可以在 child 返回并弹栈后继续生成下一个 child，不受固定 slot 数
   限制。
3. duplicate xyz 合法，因为 stack 中保存的是节点身份，不是坐标集合。
4. 无 parent、环、关闭后继续添加 child 等状态可由 grammar 直接禁止。
5. 若未来出现非 DFS 数据，再使用 parent-index pointer；当前正式数据不需要为
   0 个例外承担 pointer head 的额外复杂度。

运行时仍应保留最大节点数作为安全阀，但达到上限必须按失败计零分。stack grammar
不会自动保证模型一定生成 `CLOSE_NODE`；安全阀不能被描述为 hitmax 修复。

### 8.2 保留 UniRig 先验

- xyz token、xyz embedding、coordinate output head 和 24 层 decoder 从当前
  flat UniRig checkpoint 初始化。
- 删除 `BRANCH + parent_xyz` target，把现有 `BRANCH` token ID 重新定义为
  `CLOSE_NODE`。这样 vocabulary shape、该 token 的 embedding/head 权重和其余
  coordinate 权重都可以直接继承，不需要 resize vocabulary。
- 第一版保留 absolute xyz；parent-delta 只作为辅助 target，不替换已经验证的
  absolute coordinate head。
- 所有参数全量训练，但新模块必须先做等价或可解释初始化，不能再次冷启动一套
  Puppeteer decoder。
- 不通过打开现有 `explicit_tree_*`、`prefix_*_recovery` 或 grammar-state
  实验开关拼装新路线。stack-close 使用独立、单一、无 fallback 的 tokenizer 和
  model profile；旧 flat 路线只作为冻结对照。

### 8.3 跨层刷新 condition

仅靠序列开头的 1024 condition prefix 已被证明会在错误 rollout 中失效。第一版在
decoder 上层增加对原始 1024 condition memory 的 cross-attention adapter，并将
adapter residual 初始化为严格的 0。初始化时必须实测旧 flat prefix 的 max logit
diff 为 0；训练后允许骨架位置通过短路径重新读取 mesh，而不是穿过越来越长的错误
prefix。

condition refresh 必须通过 GT-prefix 与 self-prefix condition-swap 检验，不能只
依据模块存在或 attention weight 宣称有效。

### 8.4 训练错误 prefix，而不是只训练 GT prefix

当前 10 个 hitmax 的第一个错误全部是 coordinate；其中 6 条随后第一次结构错误都是
“GT 要 branch，模型继续 coordinate”，另外 4 条结构 action 一直到 target 末尾都
正确，但没有选择 EOS。这给出分阶段训练目标：

1. 先做 coordinate-corrupted roll-in：结构性的 stack/CLOSE prefix 保持 GT，
   但把前面 xyz 替换为模型预测或受控量化扰动，再监督下一 child、CLOSE 和 EOS。
   这直接覆盖已观测到的“坐标先偏、随后不回栈”的缺失分布。
2. 再加入少量 self-graph roll-in。发生结构偏移后，先将预测 partial tree 与 GT
   做图匹配，再监督未覆盖 child、CLOSE 和 EOS，不能把预测第 `k` 个 joint 强绑
   GT 第 `k` 个 joint。
3. 对称 sibling 使用集合目标；匹配代价考虑坐标、祖先关系和可用的 motion/skin
   evidence，不能依赖 bone-name lexical order。
4. condition-swap margin 只用于有可靠 evidence 的结构决策。低 motion 是 unknown
   evidence，不是 negative label。

### 8.5 当前结论边界

已经证明：

- 当前 conditioner 含有粗结构信息，GT prefix 下模型会使用正确 condition。
- 错误 self prefix 会逐步压过 condition，并形成不结束的 continuation attractor。
- flat parent-xyz token 在真实 valid 数据上不可逆。
- stack-close 可以精确表示当前全部 16776 棵 HGC tree，且长度预算不恶化。

尚未证明：

- stack-close 训练后一定消除 hitmax；
- cross-attention refresh 一定提高最终骨架质量；
- coordinate-corrupted 或 graph-matched roll-in 的最佳比例。

因此 stack-close 是已经通过数据 contract 的候选表示，不是已经训练成功的新模型。
下一步先实现和完成第 9 节的小规模预检，再决定是否值得正式多卡训练。

## 9. 下一次大规模训练前的硬预检

不满足以下条件，不启动 2/4 卡训练：

1. stack tokenizer 对全部正式 manifest 做 GT token/tree round-trip，并在重合
   joint 上恢复完全相同的 parent index。
2. 32 条跨 topology 小集合能同时过拟合 coordinate、重建 parent、CLOSE 和 EOS；
   不能只看 loss，必须看自由生成 J2J、topology F1 和可视化。
3. GT-prefix condition swap：正确 condition 的 coordinate、CLOSE 和 EOS NLL
   必须显著优于 swapped condition。
4. self-prefix condition swap：到 64+ token 后，CLOSE/EOS 分布不能再出现
   当前 baseline 的近零 JS collapse。
5. 生成状态机允许合法 duplicate joint，但禁止空栈 CLOSE、root 关闭后继续生成
   child，以及 root 未关闭时 EOS。
6. heldout-52 与 valid-common60 同时评测；hitmax 记为零分，不能只汇报成功行。
7. 每个评测集合都输出 GT、Prediction、Overlay，不用单一 token accuracy 代替骨架
   可用性。

### 9.1 通过预检后的正式训练候选

以下只是待讨论配置，不代表已经授权启动：

- 初始化：flat UniRig `checkpoint_sample_80000.pt`，不加载 Puppeteer decoder。
- 路线：单一 `stack_close_condition_refresh` profile，无 fallback。
- 数据：先解决 HGC 与西湖 manifest 计数差异，再冻结唯一 train/valid manifest。
- 训练：conditioner、decoder、coordinate head、stack state 和 refresh adapter
  全量训练。
- batch：维持 effective batch `48`；多卡只改变吞吐，不改变优化 batch。
- scheduler：OneCycle；旧参数与新参数的 peak LR 先由 32-row 预检确定，不能在正式
  多卡任务中边跑边猜。
- 保存与评测按 sample exposure，不按 epoch；至少在 5k、10k、40k、80k samples
  保存可复现 checkpoint。
- 每个里程碑固定跑 heldout-52 和 valid-common60；报告 all-row zero-for-hitmax
  topology F1、J2J、joint-count error、EOS/CLOSE、parent round-trip 和可视化。

## 10. 证据路径

```text
/home/wangyy/evorig/outputs/model_structure_preflight_20260719/
  condition_probe_full/condition_structure_probe.json
  condition_signal_layers_full.json
  condition_signal_layers_full.png
  condition_signal_gt_vs_self_full.json
  condition_signal_gt_vs_self_full.png
  tree_expansion_audit_v5_stack.json
  tokenizer_roundtrip_valid_randomdraw.json
  tokenizer_roundtrip_valid_fixedseed101.json
  tokenizer_roundtrip_heldout52_seed101.json
  tokenizer_roundtrip_train_seed101.json
```

已有 heldout-52 条件与前缀干预证据：

```text
/home/wangyy/evorig/outputs/matched_heldout52_20260718/diagnostics/
  flat_condition_prefix_causal_seed101_20260719.json
  flat_condition_prefix_repair_seed101_20260719.json
```

对应代码：

```text
model_training/analysis/probe_flat_condition_structure.py
model_training/analysis/analyze_flat_condition_signal_layers.py
model_training/analysis/audit_tree_expansion_feasibility.py
model_training/rigweave/scripts/audit_skeleton_token_self_consistency.py
```
