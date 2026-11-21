# MiniGPT 本机预训练 v1

这是一个从随机权重训练的英文文学续写模型。交付目标是把源数据、训练、权重恢复、独立评估和推理串起来，在 M4 MacBook Air 16 GB 上重跑。模型在固定的 27 本训练书上学习预测下一个 BPE token。

## 验收记录

| 环节 | 验收证据 |
|---|---|
| 数据 | 27 本训练书，5,994,158 个 BPE token。开发书《柳林风声》，本轮最终测试《看得见风景的房间》。来源、正文切点、SHA256 和长段落精确重复报告均已记录。 |
| 训练 | 宽度 128、4 层、20,000 次 AdamW 更新、1,024 词表。64 与 128 上下文均处理 20,480,000 个训练目标。 |
| 恢复 | 同一 40 步任务比较连续训练与从第 20 步恢复。两者验证 loss 都为 6.062033，最大参数绝对差约 0.0000612，低于检查阈值 0.001。 |
| 选择 | 《柳林风声》固定目标位置上，64 方案 loss 3.2245，128 方案读取原生上下文时 3.2390。据此选择 64 方案。 |
| 留出评估 | 新留出书同一批目标上，64 与 128 方案分别为 3.1944 和 3.2182。所选模型整书 loss 3.1890。 |
| 生成 | 权重可用相同词表加载并生成，示例见 `experiments/gutenberg_v2/generation.txt`。长段落仍有跳跃、重复与语法错误。 |

## 为什么保留 64 token

64 方案每步抽 16 个长度 64 的窗口。128 方案每步抽 8 个长度 128 的窗口。两者 token 预算和优化器步数匹配。窗口数量、位置嵌入参数与注意力计算量随之变化，所以这是两套本机训练方案的对照。

单独比较训练时的固定窗口 loss 会混入评估位置变化。`evaluate_context.py` 从同一本书取 1,024 个位置，每段只给末尾 16 个目标打分，两组回答的是同一批问题。对 128 模型还替换最早的 64 token，保留位置及近处文字。在新留出书上，真实远处文字相比替换后的 loss 为 3.2182 对 3.2229，说明该模型能利用一点远处信息。这个收益在本轮没有使 128 方案超过 64 方案。该对照使用 seed 42。模型规模与数据类型适合演示本机预训练流程，能力评估覆盖英文文学续写。

## VS Code 复现

打开仓库根目录。实验使用 Python 3.12.15、MLX 0.32.4、tokenizers 0.23.3。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python prepare_corpus.py --download
```

下载文件来自 Project Gutenberg。比较重建的 `data/gutenberg_v2/release_audit.json` 中的 SHA256 与 `experiments/context128/corpus_reference.json`。若上游版本变化，指纹就会变化。交付包包含语料配方和词表。

```bash
model=experiments/gutenberg_v2/data27_w128_l4_20k/best.safetensors
.venv/bin/python generate.py "$model" --bpe-path data/bpe_tokenizer_12books.json \
  --prompt 'The old man opened the door and' --count 48 --temperature 0.7 --top-k 20 --seed 42
.venv/bin/python evaluate_text.py data/gutenberg_v2/room_with_view_body.txt \
  --bpe-path data/bpe_tokenizer_12books.json --model context64 "$model" \
  --output experiments/context128/recheck.json
```

从随机权重重训时，在新目录保存权重，避免覆盖交付结果。将 `context=64 batch=16` 改为 `context=128 batch=8` 可重跑对照。

```bash
context=64 batch=16
run="experiments/reproduce_ctx${context}"
.venv/bin/python train_text_mlx.py --manifest data/corpus_manifest_27books.json \
  --width 128 --layers 4 --context-length "$context" --batch-size "$batch" \
  --seed 42 --steps 20000 --lr-schedule cosine --learning-rate 0.003 \
  --min-learning-rate 0.0003 --warmup-steps 300 --checkpoint-every 5000 \
  --checkpoint-dir "$run/periodic" --save "$run/final.safetensors" \
  --best-save "$run/best.safetensors" --log-file "$run/training.csv" \
  --metrics-file "$run/metrics.csv"
```

评估明细在 `experiments/context128/wind.json`、`release_test.json` 和 `release_full.json`。续训验证见同目录中的 `resume_check/comparison.json`。在 M4 上，历史 64 和本轮 128 的实测训练时间分别约 311 和 286 秒，MLX 峰值分别为 177.7 和 256.4 MB。这些是各自运行观测，整机内存和速度会受环境影响。

## v1 结束条件

从语料重建到训练、恢复、生成和留出测试的流程均有日志或权重可复核。后续如要做指令问答、业务任务或更大规模训练，先为那个目标单独设评估和预算。
