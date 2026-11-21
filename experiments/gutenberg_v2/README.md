# MiniGPT 27 本书实验

## 先读这份记录

十二书版本已经覆盖了从零预训练到保存权重、恢复训练、生成和评估的流程。它的三组随机种子适合用来判断同一训练方案的波动。本轮沿用宽度 64、2 层、上下文 64、1,024 个 BPE token 的 seed 42 权重作为对照。旧版上下文 32 和 64 在同样步数下分别预测 512 和 1,024 个 token/步，比较的是两套训练方案的整体表现。

## 27 本从哪来

旧清单里的 12 本保留，新增 15 本 Project Gutenberg 英文作品，两批加起来是 27 本。《双城记》《简·爱》《白鲸》和《德古拉》是其中四本。`data/gutenberg_v2/sources.json` 记录新增书号和正文起点，`data/gutenberg_v2/audit.json` 记录每个文件的 SHA256。原始文本叫 `*_raw.txt`，从正文起点截取并去除 Gutenberg 结尾后的训练文本叫 `*_body.txt`。

- 训练材料增加到 5,994,158 个 BPE token，十二书基线为 1,473,621 个。
- 旧的 1,024 项 BPE 词表沿用。这样扩数据时，`the` 等子词的 ID 和验证题保持一致。
- 开发集继续使用《柳林风声》，供训练过程保存最佳权重和比较方案。
- 新下载的《米德尔马契》留作最终测试，独立于 27 本训练书。Anne 和 Sherlock 保留作历史结果。
- 审计把整书和至少 40 词的完整段落转成忽略大小写、标点与空白的形式做精确比较。当前无整书重复或跨书长段落精确重复。短摘录、改写和段落边界变化仍可能漏过。

## 三个实验问题

| 比较 | 保持相同 | 本次变化 |
|---|---|---|
| 十二书 → `data27_small` | 种子、模型、训练步数、词表、验证书 | 改变训练语料来源和覆盖范围 |
| `data27_small` → `data27_w128_l4` | 27 本训练书、种子、训练步数、词表、验证书 | 宽度 64→128，Transformer 块 2→4，整体容量一起变 |
| `data27_w128_l4` → `data27_w128_l4_20k` | 数据、模型、种子、词表、验证书、学习率上下限 | 从头跑两倍步数，余弦衰减也拉长 |

每个更新抽取 16 个窗口，每个窗口给模型 64 道“预测下一个 token”的题。10,000 步处理 10,240,000 个目标位置，20,000 步处理 20,480,000 个。窗口会随机重复，处理量包含重复见过的 token。宽度就是一个 token 的内部向量有多少维，层数是它依次经过多少个 Transformer 块。增加宽度和层数后模型从 236,288 个参数增加到 1,064,704 个参数，同时增加了每步的计算成本。

## 两种评估

训练过程中每 100 步检查《柳林风声》的 64 个固定窗口。`best.safetensors` 保存窗口 loss 最低的时刻，`final.safetensors` 保存最后一步。`*.optimizer.safetensors` 保存同名权重的 AdamW 状态。

`evaluate_text.py` 用滑动窗口给整本书的每个目标 token 打一次分，首 token 作为起始上下文。全书 loss 是答案的平均负对数概率，较低代表模型通常给正确 token 更高的概率。固定跨模型上下文 64、步幅 16 和词表后可以直接比较同一本书。

另选 64 段做续写排序。每题给相同的前缀，比较正确的 16 token 续写与交换中间两个 4-token 片段的版本。两份答案的 token 一样，只变顺序，单靠词频只能打平。看正确版本得分更高的次数、平局数和得分差。任务衡量模型对词序和局部上下文的敏感度，开放续写需要再用生成样例观察。

## 实测结果

| 训练方案 | 参数量 | 最佳权重步数 | Wind 固定窗口 loss | Wind 全书 loss | Wind 排序 |
|---|---:|---:|---:|---:|---:|
| 12 本，64 宽 2 层，10,000 步 | 236,288 | 9,600 | 3.7568 | 3.6122 | 61/64 |
| 27 本，64 宽 2 层，10,000 步 | 236,288 | 9,400 | 3.6532 | 3.5442 | 62/64 |
| 27 本，128 宽 4 层，10,000 步 | 1,064,704 | 9,900 | 3.4454 | 3.3144 | 62/64 |
| 27 本，128 宽 4 层，20,000 步 | 1,064,704 | 20,000 | 3.3409 | 3.2264 | 64/64 |

按 Wind 全书 loss 选最后一组。然后在新测试书《米德尔马契》的同 685,211 个目标 token 上比较十二书基线与所选权重。平均 loss 从 3.7004 降到 3.2469。两组真实续写排序都为 64/64，这个任务的正确数已经饱和。模型给真实续写的相对优势从平均 1.759 增到 2.168 nat/token。这些结果只反映当前文学语料与评估题。各新配置只跑 seed 42。

生成记录见 `generation.txt`。提示 `The old man opened the door and` 上，所选模型贪心接出 `looked at me.`，后面的对话出现不连贯。整书 loss 能衡量真实文本概率，固定提示则展示自由生成的实际表现。

20,000 步训练耗时约 5 分 11 秒，训练脚本报告 MLX 峰值 177.7 MB。原计划的宽度 192、4 层配置启动时单头注意力需要 40,448 字节的 Metal 线程组内存，高于本机上限 32,768 字节，已保留失败日志 `data27_wide/train.log`。成功实验使用宽度 128、4 层。

## 在 VS Code 复现

打开 `mini-gpt` 文件夹，在其集成终端执行。语料清单基于已有十二书文件和 tokenizer。

```bash
.venv/bin/python prepare_corpus.py --download

run=experiments/gutenberg_v2/data27_w128_l4_20k
.venv/bin/python train_text_mlx.py --manifest data/corpus_manifest_27books.json \
  --steps 20000 --seed 42 --width 128 --layers 4 --context-length 64 \
  --learning-rate 0.003 --min-learning-rate 0.0003 --lr-schedule cosine \
  --warmup-steps 300 --checkpoint-every 5000 --checkpoint-dir "$run/periodic" \
  --best-save "$run/best.safetensors" --save "$run/final.safetensors" \
  --run-name data27_w128_l4_20k \
  --log-file experiments/gutenberg_v2/training.csv \
  --metrics-file experiments/gutenberg_v2/metrics.csv
```

重复训练时可改输出文件路径，避免覆盖已保存的权重或在同一 CSV 中混入同名实验。

```bash
.venv/bin/python evaluate_text.py data/gutenberg_v2/middlemarch_body.txt \
  --bpe-path data/bpe_tokenizer_12books.json \
  --model base12 experiments/checkpoints/seeds_width64_context64/seed42/best.safetensors \
  --model candidate experiments/gutenberg_v2/data27_w128_l4_20k/best.safetensors \
  --output experiments/gutenberg_v2/middlemarch_recheck.json
```
