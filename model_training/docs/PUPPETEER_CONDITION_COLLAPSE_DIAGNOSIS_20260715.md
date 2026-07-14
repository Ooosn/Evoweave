# Puppeteer 条件坍塌诊断

状态：2026-07-15 已在 HGC 单张 H100 上完成因果验证。

## 结论

此前 Puppeteer 路线生成固定错误姿态，不是 GT 读成 reset pose，也不是训练与生成代码错位。直接原因是：

1. 可训练的 motion conditioner 把同一资产的不同 pose 编码成近乎相同的条件 token，主要保留资产身份，丢掉 pose。
2. 标准 teacher forcing 会把当前 GT pose 的历史坐标 token 放进前缀。
3. decoder 因而可以忽略 condition，改从 GT 前缀恢复后续 pose。
4. 自由生成没有 GT 前缀，只能从一个模态/模板姿态开始，随后自洽地生成整套错误骨架。

这是一条经干预验证的因果链，不是根据低 F1 做出的猜测。

## 定义

- **query pose**：一次数据读取随机选择的目标帧。输入 mesh 和监督骨架都来自这同一帧。
- **condition tokens**：surface tokenizer 和 motion encoder 输出的 1024 个、送入 AR decoder 的条件向量。
- **teacher-forcing prefix**：训练时，在预测当前位置 token 前喂给 decoder 的 GT 历史 skeleton token。
- **free generation**：推理时只从 BOS 开始，后续前缀全部由模型自己的预测组成。
- **condition collapse**：不同 query pose 对应的 condition tokens 在数值上趋于相同，无法再可靠区分姿态。
- **condition swap**：保持同一 GT 前缀，只把 pose A 的 condition 换成 pose B，用于测 condition 对 logits 的真实影响。
- **prefix swap**：保持同一 condition，只把 teacher-forcing 前缀换成另一个 pose，用于测 GT 前缀泄露了多少 pose 信息。

## 已排除项

- 输入 mesh 与 target skeleton 使用相同的随机 query frame；不存在固定用 reset-pose GT 的读取错误。
- teacher forcing 与逐 token generation 在相同前缀上的 logits 已对齐；不是训练/推理位置偏移。
- pre-LN 的随机初始化 decoder 能拟合，说明不是必须依赖 Puppeteer 预训练权重。
- joint-slot / target-aware embedding 不是这次固定错误姿态的根因。
- 单纯增加训练步数不能解决；可训练 conditioner 的 pose collapse 会随训练加重。

## 对照实验

共同设置：32 个 train 资产、8 个 valid 资产、随机 query pose、identity-1024 条件、随机初始化 24 层 pre-LN decoder、batch size 1。唯一主要变量是 conditioner 是否训练。

| 指标（8 个成对 pose 的中位数） | conditioner 可训练 step1000 | conditioner 可训练 step5000 | conditioner 冻结 step1000 | conditioner 冻结 step3000 |
|---|---:|---:|---:|---:|
| condition 相对 L2 | 0.0191 | 0.00490 | 1.247 | 1.251 |
| condition cosine | 0.9998 | 0.99999 | 0.222 | 0.218 |
| condition swap 的 logit 相对 L2 | 0.000725 | 0.00142 | 0.0528 | 0.0874 |
| prefix swap 的 logit 相对 L2 | 0.0289 | 0.145 | 0.0339 | 0.0896 |
| condition/prefix logit 影响比 | 0.0245 | 0.0108 | 1.43 | 0.866 |
| condition swap 的 argmax 变化率 | 0% | 0% | 12.9% | 18.6% |

可训练路线到 step5000 时，GT token 在两个 pose 间改变 32% 到 63%，但 7/8 资产的 condition 几乎完全相同。相同 GT 前缀下替换 condition，decoder argmax 基本不变；替换 GT pose 前缀则会改变约 28% 的 argmax。

冻结 conditioner 后，condition 与 GT prefix 对 logits 的影响恢复到同一量级，证明不是 decoder 天生无法使用 pose，而是联合训练找到了 teacher-forcing shortcut。

## 自由生成证据

可训练 conditioner：

- step1000 与 step5000 的前 4 个 train 资产，在两种明显不同 pose 下，4/4 生成 token 序列完全相同。
- step5000 的两个大骨架都退化为同一棵 4-joint 树。

冻结 conditioner，step3000：

- 前 4 个 train 资产 joint 数全部正确，全部正常 EOS，0 hitmax。
- 两种 pose 的平均 topology F1 为 0.75/0.80，平均 J2J 约 0.024。
- 3/4 资产的生成 token 随 pose 改变；剩余资产的两帧本身只有 1/4 joint 变化。
- 8/8 valid 资产的生成 token 都随 pose 改变，全部正常 EOS，0 hitmax。
- valid 平均 topology F1 仍只有 0.12/0.16，且常过生成 joint；这是只训练 32 个资产的泛化限制，不能把该消融当成最终模型质量。

## 修复边界

`freeze_conditioner` 是根因消融和可用的阶段一训练方式，但不是最终“全量微调”方案。它证明了保留 pose condition 可以消除固定姿态故障；它没有证明应该永久冻结一个随机 motion encoder。

正式全量微调至少需要同时满足：

1. condition tokens 必须有直接的 query-geometry 保真约束，例如逐 anchor 重建当前 query point，或不可被 motion encoder 抹掉的几何 skip。
2. decoder 不能只在泄露完整 GT pose 的坐标前缀上学习；需要让一部分训练状态使用模型坐标前缀，同时保持合法 tree/topology contract。
3. 训练期间必须持续运行 paired-pose audit。若 condition cosine 再次接近 1，或 condition swap 的 argmax 变化回到 0，应立即判定路线失败，而不是继续烧卡。

在上述约束落地前，不应再次提交“conditioner 全量可训练 + 纯 teacher forcing”的正式 Puppeteer 任务。

## 证据位置

HGC：

```text
/home/wangyy/evorig/diagnostics/puppeteer_hgc_identity1024_preln_motion_decoder_32asset_5000step_6dd6d30
/home/wangyy/evorig/diagnostics/puppeteer_hgc_identity1024_preln_frozen_conditioner_32asset_3000step_1bd9785
```

本地可视化与 JSON：

```text
D:\evoweave\outputs\puppeteer_hgc_oneh100_20260715
```

诊断实现：

```text
model_training/analysis/diagnose_puppeteer_teacher_forcing.py
```
