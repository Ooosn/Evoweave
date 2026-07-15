# Puppeteer 条件坍塌因果诊断

状态：2026-07-15 已在 HGC 单张 H100 上完成配置还原、受控重训和 2x2 消融。

## 结论

此前正式 Puppeteer 任务不是单一故障，而是两个相互独立的问题：

1. **解码器错误继承了 post-LN 配置。** 随机初始化的 24 层 SkeletonOPT 在该配置下几乎学不动。UniRig 使用 pre-LN。
2. **learned-query projector 把 1024 个条件 token 压成 257 个匿名槽位。** 把解码器改成 pre-LN 后，网络虽然开始拟合 teacher-forcing 目标，但 projector 与 conditioner 会把不同 pose、不同 mesh 映射成近似相同的 condition。

旧实验中，持续使用恒定 `1e-4` 也会使 direct-1024 路线坍塌；但真正的四卡任务使用了 OneCycle，所以学习率不是那次任务“完全学不动”的首要原因。学习率问题、norm 配置问题和 projector 问题必须分开描述。

## 正式失败任务的真实配置

历史任务 `evoweave_jointtoken_hxr4gpu_20260709_878622c` 对应提交 `878622c`，其关键配置为：

- 随机初始化 24 层 OPT-350M 规格解码器；
- `do_layer_norm_before=false`，即 post-LN；
- `condition_projection=cross_attention`；
- 1024 个 motion-condition token 被 learned queries 压成 257 个 token；
- surface tokenizer、motion encoder、projector 和 decoder 全量训练；
- 所有参数组峰值学习率 `1e-4`，OneCycle；
- 四卡有效 batch size 为 48，共 1667 optimizer steps。

因此，不能再用旧的 identity-1024 恒定学习率实验替代这次正式任务的根因分析。

## 因果实验一：post-LN

两组实验均使用 32 个训练资产、1000 steps、cross-attention-257、surface tokenizer 可训练、OneCycle，唯一变化是 norm 位置。

| 配置 | step 1000 token accuracy | step 1000 parent accuracy | 结论 |
|---|---:|---:|---|
| post-LN | 0.051 | 0.000 | 基本没有学会目标序列 |
| pre-LN | 0.416 | 1.000 | 解码器能够快速拟合 |

只改一项就从 parent accuracy 0 变成 1，因此 post-LN 是“训练后仍像随机模型”的直接原因。该结论不依赖 Puppeteer 预训练权重。

## 因果实验二：257-token projector

在 pre-LN、OneCycle 下，对 `condition projection` 和 `surface tokenizer 是否冻结` 做 2x2 对照。表中的距离都是训练后 conditioner 输出的相对差异；越接近 0，表示不同输入越难区分。

| condition 路线 | surface | 同资产不同 pose | 不同资产 | 结果 |
|---|---|---:|---:|---|
| cross-attention-257 | 可训练 | 0.0244 | 0.0509 | 严重坍塌 |
| cross-attention-257 | 冻结 | 0.0281 | 0.0855 | 仍然坍塌 |
| identity-1024 | 可训练 | 1.2587 | 1.4418 | 保留 mesh/pose 差异 |
| identity-1024 | 冻结 | 0.5709 | 0.8459 | 保留 mesh/pose 差异 |

所以 surface tokenizer 是否更新不是主因。决定性变量是 learned-query 257-token projector：

- 257 个 learned queries 没有与 1024 个 mesh anchors 的一一对应关系；
- cross-attention 可以退化为对全体 anchors 的相似 pooled summary；
- teacher forcing 允许 decoder 依赖 GT skeleton prefix，因而没有损失项强迫这个 pooled condition 保留 query geometry；
- 梯度随后会共同推动 projector 和 motion encoder 形成近常量 soft prompt。

identity-1024 不做压缩、不换序，也没有 learned query；每个 condition token 仍对应原来的 surface anchor，因此不会通过这个接口丢掉空间身份。

## 学习率与 scheduler

学习率确实是风险，但不是对正式四卡故障的完整解释：

| direct-1024 配置 | 同资产不同 pose | 跨资产距离 | 结论 |
|---|---:|---:|---|
| 恒定全量 `1e-4`，step 5000 | 0.00124 | 0.00476 | optimizer 将 conditioner 训塌 |
| motion LR `1e-5` | 0.2783 | 0.8912 | 明显缓解 |
| OneCycle，峰值全量 `1e-4` | 0.5709 | 0.8459 | 没有坍塌 |

这说明“峰值到过 `1e-4`”不是充分原因；长时间维持全量 `1e-4` 才是旧 direct-1024 实验的致因。当前正式 baseline 因此固定 OneCycle，同时继续记录 condition separation。

## 为什么 UniRig 没有同样失败

UniRig 路线同时具备：

- pre-LN decoder；
- 1024 个条件 token 直接进入 decoder，不经过 1024→257 learned-query 压缩；
- OneCycle scheduler；
- 已验证的静态 skeleton decoder 初始化。

预训练初始化会帮助收敛，但不是上述因果结论成立的前提。随机初始化的 pre-LN 控制组已经能学习；随机初始化的 post-LN 控制组几乎不能学习。

## 固定后的 baseline contract

正式 joint-token baseline 现在必须满足：

- `query_tokens=1024`；
- `cond_length=1024`；
- `condition_projection=identity`；
- `decoder_norm_style=pre`，且 resolved config 的 `do_layer_norm_before=true`；
- joint-slot embedding 开启；
- train random query pose 开启；
- scheduler 为 OneCycle。

训练入口默认启用 `--require-query-preserving-baseline-contract`。任一条件不满足会在构建模型前直接终止，不能再静默启动正式任务。`training_contract.json` 会保存这些 resolved 字段。

## 修复后真实数据 preflight

在 HGC 单张 H100、rootless-v3 HGC manifest 上重新构建随机初始化的完整 24 层模型，得到：

- resolved condition shape 为 `[1, 1024, 1024]`，projection 为 identity，decoder 为 pre-LN；
- 随机 query 为 frame 8，不是 frame 0；mesh-query 和 target-query 最大绝对误差均为 `0`；
- 当前 target 与 frame 0 target 的 RMS 差异为 `0.12856`，排除了固定 reset-pose GT；
- 旋转 query 后 condition relative L2 为 `0.58355`，初始化路径没有条件坍塌；
- teacher forcing 与逐 token generation 对相同 prefix 检查 32 个位置，max logit difference 为 `0`；
- 反向传播中 surface、motion、joint-slot embedding、decoder blocks 和 decoder token path 均收到有限的非零梯度；identity projector 本身没有参数，因此该组梯度元素为 0 是预期行为；
- 没有未分配的可训练参数。

完整 condition/pose/contract audit 与 gradient audit 分开执行，以保持单卡诊断显存有界：

```text
/home/wangyy/evorig/diagnostics/puppeteer_fixed_contract_full_preflight_20260715
/home/wangyy/evorig/diagnostics/puppeteer_fixed_contract_gradient_preflight_20260715
```

## 尚未被证明的部分

上述实验已经证明旧实现为何学不动、为何会丢掉 mesh/pose 条件，也证明新 contract 能学习并保留 condition 差异；它还没有证明全量自由生成质量已经达标。下一次正式训练仍需用固定失败样本做 greedy generation、拓扑 F1、joint count 和 condition-swap 检查，不能只看 teacher-forcing accuracy。

## 证据位置

```text
/home/wangyy/evorig/diagnostics/puppeteer_causal_formalroute_crossattn257_onecycle_all1e4_32asset_1000step_20260715
/home/wangyy/evorig/diagnostics/puppeteer_causal_formalroute_crossattn257_preln_onecycle_all1e4_32asset_1000step_20260715
/home/wangyy/evorig/diagnostics/puppeteer_causal_crossattn257_preln_frozensurface_onecycle_32asset_1000step_20260715
/home/wangyy/evorig/diagnostics/puppeteer_causal_identity1024_preln_trainablesurface_onecycle_32asset_1000step_20260715
/home/wangyy/evorig/diagnostics/conditioner_separation_unirig_best_val_20260715.json
```

诊断实现：

```text
model_training/analysis/diagnose_dynamic_conditioner_separation.py
model_training/analysis/diagnose_puppeteer_teacher_forcing.py
```
