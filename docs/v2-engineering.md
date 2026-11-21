# MiniGPT v2 工程说明

## 目标与范围

交付一个可复现的英文文学语言建模项目。数据准备、训练、恢复、模型选择、独立测试、推理优化和本机接口均提供实际命令及证据。MLX v1 的训练入口和权重继续保留。v2 使用 PyTorch，CPU 与 MPS 已在本机验证，CUDA 路径由设备自动选择，需在 NVIDIA 环境另行验证。

## 训练设计

- Pre-LayerNorm decoder、多头因果注意力、PyTorch SDPA、GELU FFN 和共享输入输出嵌入。
- 在文档内部随机取窗口，按有效窗口数加权采样，避免跨书拼接窗口。训练前检查路径交集、整文件重复和数据指纹。词表沿用旧训练语料训练的 1,024 项 BPE。
- AdamW、100 步 warmup、cosine 衰减、全局梯度裁剪、梯度累积。loss 和梯度非有限时终止。当前使用 float32，便于本机可重复验证。
- checkpoint 包含模型、优化器、CPU 采样器、CPU RNG、步数、最佳验证分数与协议。单文件采用临时文件再替换，防止部分写入。恢复校验 tokenization、语料和训练配置。
- CPU 连续训练与断点恢复逐参数相等。MPS/CUDA 跨硬件逐位一致性需额外验证。本实现没有 dropout，采样使用显式 CPU generator。

## 模型选择

实验计划在 configs/experiment_plan.json。small、medium 与 large 各自训练 20,000 次、每次 1,024 个目标 token，均为 seed 42。按照 Wind 固定验证窗口最低 loss 选定 checkpoint 和方案。这些方案同时改变宽度、深度与头数，解释为完整配置对照。

旧 v1 测试书继续用作回归材料。最初保留《红字》作本轮评估材料，正文来自大学镜像中的 Gutenberg #33，切掉引言与 Gutenberg 页脚，保留数据来源和 SHA256，并检查与训练书的 40 词精确重复。该检查覆盖长段落原样重复，短语、近似改写及语义重叠仍需其他方法。

完整文本评分使用上下文 64、步幅 16 的滑动窗口，每个目标 token 恰好评分一次，报告 token 数、平均交叉熵及 perplexity。公开任务和服务验收见 [指标口径](industry-metrics.md)。不同评分协议的 loss 分开报告。

## 推理与交付

- KV-cache 保存逐层历史 K/V。单 token decode 可读取全部已缓存历史，prefill 使用因果 mask。
- 学习式位置嵌入达到上下文上限后，重建最后 context 个 token 的缓存，保持与旧滑动窗口语义一致。缓存与无缓存 logits 和溢出后的贪心序列均有回归测试。
- export 导出推理权重、配置、词表、训练来源和 SHA256，移除优化器状态。加载时核对词表指纹和模型配置。
- HTTP 服务绑定 localhost，提供 /health 和 /generate，限制请求体、输入长度和生成长度。并发生成时返回 429。适用于本机集成验证。外部上线还需网关、身份认证、TLS、超时控制、请求队列与监控。

## 复现

~~~bash
python3.12 -m venv .venv-torch
.venv-torch/bin/python -m pip install -r requirements-torch.txt
.venv-torch/bin/python prepare_corpus.py --download
.venv-torch/bin/python -m unittest discover -s tests -v
.venv-torch/bin/python -m minigpt_v2.train \
  --manifest data/corpus_manifest_27books.json \
  --config configs/v2-medium.json --output experiments/my_medium --steps 20000
~~~

Windows 的 VS Code 中使用虚拟环境 Scripts 目录里的 python.exe。训练入口自动选 CUDA、MPS 或 CPU。

同一计划中断恢复，steps 仍填原定总步数：

~~~bash
.venv-torch/bin/python -m minigpt_v2.train \
  --manifest data/corpus_manifest_27books.json \
  --config configs/v2-medium.json --output experiments/my_medium --steps 20000 \
  --resume experiments/my_medium/last.pt
~~~

更换语料和训练计划应另建实验目录。断点恢复和权重续训分别使用 --resume 与 --init。


## 权重续训与断点恢复的区别

断点恢复沿用同一优化器、采样状态和学习率计划。另一种情况是已完成预训练后开启一个新的低学习率阶段，此时用 --init 指定 v2 模型目录，并创建新实验目录。例如：

~~~bash
.venv-torch/bin/python -m minigpt_v2.train \
  --manifest data/corpus_manifest_27books.json --config configs/v2-medium.json \
  --init artifacts/v2 --output experiments/continued --steps 5000 \
  --learning-rate 0.0003 --min-learning-rate 0.00003
~~~

新阶段记录初始化权重的 SHA256，并核对模型结构与词表。它重建优化器和采样器。额外训练消耗的目标 token 应单独计入预算，再由开发集比较是否改善。


## 长上下文迁移

最终版本由段落清洗版本扩展至 256 token 上下文。--extend-context 显式允许位置表扩展，要求宽度、层数、头数和词表一致。新增位置使用旧位置表周期平铺，已有位置保持原值。短前缀等价性由测试验证。5,000 步迁移训练后，用共同的验证目标选择版本。公开基准用于诊断的先后顺序见 TRAINING_GUIDE.md。
