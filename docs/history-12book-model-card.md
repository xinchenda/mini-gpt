# MiniGPT 十二书模型卡

## 模型概览

- 类型：使用 Byte-Level BPE 子词词表的 Transformer 语言模型
- 参数量：236,288
- BPE 词表：1,024
- Transformer 层数：2
- 隐藏宽度：64
- 上下文：64 token
- 训练更新：10,000 步
- 优化器：AdamW
- 学习率：warmup 300 后余弦衰减，0.003 到 0.0003

## 数据

训练集包含 12 本 Project Gutenberg 英文作品，训练时拼接后共 3,815,905 字符和 1,473,621 个 BPE token。

- 开发验证：Wind in the Willows
- 历史测试：Anne of Green Gables
- 新最终测试：The Adventures of Sherlock Holmes

数据角色记录在 `data/corpus_manifest_12books.json` 和 `data/corpus_manifest_12books_sherlock.json`。

## 结果

| 指标 | 三种子均值 | 样本标准差 |
|---|---:|---:|
| Wind 最佳 loss | 3.7433 | 0.0133 |
| Anne loss | 3.6497 | 0.0046 |
| Anne 估算 bit/字符 | 2.0979 | 0.0027 |
| Sherlock loss | 3.5850 | 0.0113 |
| Sherlock 估算 bit/字符 | 2.0218 | 0.0064 |

以上 loss 使用固定抽样窗口。bit/字符由抽样 loss 与整书 token/字符比例折算。

## 推荐权重

三个种子的最佳权重位于：

- `experiments/checkpoints/seeds_width64_context64/seed42/best.safetensors`
- `experiments/checkpoints/seeds_width64_context64/seed43/best.safetensors`
- `experiments/checkpoints/seeds_width64_context64/seed44/best.safetensors`

对应的 `.optimizer.safetensors` 包含 AdamW 状态。

## 生成命令

```bash
.venv/bin/python generate.py \
  experiments/checkpoints/seeds_width64_context64/seed44/best.safetensors \
  --prompt "To Sherlock Holmes she is" \
  --count 80 \
  --temperature 0.7 \
  --top-k 20
```

## 限制

- 模型规模属于教学级，不能作为通用语言模型。
- 训练数据仅来自 12 本英文书籍。
- 评估主要是语言建模 bits/char，不包含下游任务 benchmark。
- 当前训练只使用单机 MLX，没有分布式训练。
- 生成文本可能出现重复、虚构单词和事实错误。
