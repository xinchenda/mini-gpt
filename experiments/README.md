# 实验文件说明

- `summary.csv`：每次训练一行的汇总结果。
- `metrics.csv`：第 0 步以及每 100 步评估一次的曲线数据。
- `test_results.csv`：模型固定后执行的一次性独立测试结果。
- `seed_summary.csv`：多种子训练的均值、标准差和范围。
- `paired_seed_comparison.csv`：宽度 64 与 128 的配对种子逐项结果。
- `final_seed_summary.csv`：三个选定种子在 Anne 测试集上的汇总。
- `context_seed_comparison.csv`：上下文 32 与 64 的配对种子结果。
- `final_test_seed_comparison.csv`：Anne 与 Sherlock 的三个种子最终测试结果。
- `checkpoints/pytorch/`：最早的 PyTorch checkpoint。
- `checkpoints/baseline/`：早期字符词表、目录清理和元数据格式实验。
- `checkpoints/width32/`：宽度 32 的单层模型。
- `checkpoints/width64_layers1/`：宽度 64 的单层模型。
- `checkpoints/width64_layers2/`：宽度 64、2 层的恒定学习率实验。
- `checkpoints/width64_layers2_warmup/`：宽度 64、2 层、warmup 实验。
- `checkpoints/crossbook/`：使用 Alice 训练、独立文本验证的实验。
- `checkpoints/multibook/`：使用 Alice 和 Looking-Glass 两本书训练、Oz 独立验证的实验。
- `checkpoints/bpe/`：BPE tokenizer 的模型检查点。
- `checkpoints/corpus5/`：五本书训练、Wind 开发验证的字符级和 BPE 检查点。
- `checkpoints/corpus8/`：八本书训练、Wind 开发验证的 BPE 检查点。
- `checkpoints/corpus12/`：十二本书训练、Wind 开发验证的 BPE 检查点。

`summary.csv` 是判断实验配置和最终结果的入口。新 checkpoint 会在同目录保存同名 `.optimizer.safetensors`，其中包含 AdamW 一阶矩、二阶矩和优化器步数。元数据还记录 Python 随机采样状态。

当前推荐使用的权重：

`checkpoints/width64_layers2_warmup/layers2_warmup_10000_best.safetensors`
