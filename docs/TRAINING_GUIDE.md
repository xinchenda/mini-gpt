# MiniGPT 训练与交付记录

## 阅读入口

项目任务是英文文本的自回归续写。训练经历了文学语料预训练、输入格式修正、上下文迁移和百科语料扩展。输入一段文字，模型逐 token 生成后续内容。本机版本包含完整训练配方、模型包、评测记录和本机推理服务。云端 98M 版本已完成 20 亿累计目标训练，状态和费用见 [云端训练记录](CLOUD_TRAINING.md)。

- 快速判断效果，读 [结果与验收](results.md)。
- 检查模型结构、数据和适用范围，读 [模型卡](../MODEL_CARD.md)。
- 理解实现取舍，读 [工程说明](v2-engineering.md)。
- 核对公开基准与服务指标，读 [指标口径](industry-metrics.md)。

## 1. 项目演进

| 阶段 | 工作内容 | 代码或证据 |
| --- | --- | --- |
| 梯度与优化器 | 标量平方损失，手算梯度，观察学习率过大时的变化 | examples/gradient_step_mlx.py |
| 字符级 GPT | 用动物句子学习输入与目标错位、因果注意力和自回归生成 | train.py、train_mlx.py |
| 真实文本 | 从 Alice 单书扩展到多本书，分离验证文本，比较字符词表与 BPE | train_text_mlx.py |
| 十二书实验 | 学习率调度、梯度裁剪、宽度/层数/上下文对照、多种子统计 | experiments/history、[历史模型卡](history-12book-model-card.md) |
| MLX v1 | 27 本书、20,000 步；同预算上下文对照、恢复验证和独立文本评估 | [v1 验收](v1-release.md) |
| PyTorch v2 | 多头注意力、文档内采样、完整状态恢复、容量与采样对照、模型导出和 HTTP | minigpt_v2、tests、experiments/v2 |
| 段落格式修正 | 从公开基准失败定位硬换行偏置，用共同验证集进行清洗对照 | normalize_corpus.py、format_ablation.json |

前期历史实验的词表、上下文和评测位置不同，数值按各自协议保存。最新模型的比较统一使用本轮记录的协议。

## 2. 数据准备与分离

27 本训练书来自 Project Gutenberg。原文下载地址、正文起点和书号位于 data/legacy_sources.json 与 data/gutenberg_v2/sources.json。准备脚本去掉 Gutenberg 页眉页脚并裁切正文，生成语料 manifest，保留来源和 SHA256。

训练前校验角色文件互不重叠、整文件内容没有重复。文档采样按有效窗口数加权，窗口保持在单本书内。新增留出书检查与训练书的连续 40 词精确重复。该检查的范围是原样长片段，短引用和近似改写另行处理。

v1 词表使用训练书拟合，v2 沿用同一份 1,024 项 Byte-Level BPE。词表文件随模型包发布，加载和恢复时检查 SHA256。

### 排版硬换行

原书把长段落按固定行宽换行。原版模型在有同类排版的测试文本上取得正常 loss，但迁移到连续自然段落后，出现强换行偏置，LAMBADA 最后一个词的准确率为零。

修正方法是在同一段落内部连接换行，段落间保留空行。Romeo and Juliet 以诗剧形式保留原有行边界。原始文本和处理后文本分别存放，处理前后的指纹和换行数量写入 audit.json。

原训练格式和段落清洗格式从相同权重开启 5,000 步训练，学习率、seed、每步目标数相同，使用同一份清洗后的 Wind 验证文本。数据清洗的收益由这个对照判断。

最初留出的《红字》已经用于诊断回归。《七个尖角阁的宅第》用于格式修正后的回归评估。《化身博士》用于上下文方案的回归评估。最后的语料扩展决策后另行预留《失落的世界》，在最终版本确定后评分。公开 LAMBADA 曾用于发现问题，其最终成绩按测试知情开发的结果解释。

## 3. 模型结构

PyTorch v2 使用 Pre-LayerNorm decoder、多头因果 SDPA、4 倍宽度的 GELU 前馈网络，以及共享的输入输出嵌入。三个容量候选保持 64 token 上下文与 1,024 项词表。

| 配置 | 宽度 | 层数 | 注意力头数 | 参数量 |
| --- | ---: | ---: | ---: | ---: |
| small | 128 | 4 | 4 | 932,608 |
| medium | 192 | 6 | 6 | 2,878,464 |
| large | 256 | 6 | 8 | 5,017,600 |

容量对照改变了宽度、深度和头数，结论针对完整配方。该组实验可回答同等训练预算下哪种配置更合适。

## 4. 训练过程

每步使用 16 个长度 64 的输入窗口，预测 1,024 个目标 token。三个容量候选各完成 20,000 次 AdamW 更新，目标预算为每种配置 20,480,000 token。small 额外运行一个随机种子用于检查波动。

基础学习率由 100 步 warmup 升到 0.001，随后 cosine 衰减到 0.0001。梯度范数上限为 1。训练提供梯度累积、非有限 loss/梯度检查和逐检查点日志。

checkpoint 保存模型、优化器、采样器、CPU RNG、步数、数据指纹和训练协议。写文件使用临时文件替换。CPU 回归测试验证完整训练与中断恢复的逐参数一致性；修改语料或原计划参数时，恢复入口拒绝继续。

## 5. 优化实验与选择

1. 用固定的 Wind 验证窗口选择每个容量方案的 checkpoint。
2. 对 medium 和 large 做 5,000 步低学习率续训，最高 0.0003，末值 0.00003。
3. medium 的混合采样采用 70% 长度比例加 30% 书籍均匀权重，与普通采样做同初始化、同预算对照。首轮差距小于 0.001 nat/token，稳定增益仍需更多重复实验。
4. 对 large 做原格式/段落清洗对照，共同的验证集决定数据格式。
5. 将清洗版本上下文扩展至 256 token。旧位置向量周期平铺初始化新增位置，短前缀的初始 logits 保持一致，再训练 5,000 步。两个版本读取相同的 256-token 验证窗口，只给相同的末尾 16 个目标打分；短模型仅看到最近 64 token。
6. 增加官方 WikiText-2 训练切分，文学与 WikiText 独立验证切分共同选择通用版本。WikiText 测试切分单独评分，数据来源和指纹写入 audit.json。
7. 版本锁定后，对新的留出书、公开任务及推理服务评分。

完整配置见 configs/experiment_plan.json，日志与选择结果位于 experiments/v2。模型选择由开发集完成。公开任务曾暴露输入格式问题，相关开发顺序在记录中保留。

## 6. 质量评测

完整文本评分使用各模型原生上下文（v1 为 64、v2 为 256）、步幅 16 的滑动窗口，每个目标 token 评分一次。记录文本 SHA256、目标数、平均交叉熵、token PPL、按空白词计的 word PPL 和 bits/byte。

- 新留出书验证同类文学文本的泛化。
- WikiText-2 测试文本验证跨领域语言建模。保存原始测试文件的来源及指纹。
- 完整 LAMBADA OpenAI 5,153 题验证最后一个词的预测，使用完整目标词的贪心准确率和 word PPL。
- 固定提示词生成样例保留重复和语义错误，描述性重复率与文本一起交付。

与论文结果比较时核对文本版本、预处理、词表、上下文及输出约束。公开论文数字列为参考；本项目 v1/v2 的同口径实测结果直接并列。

## 7. 推理与模型包

KV-cache 保存每层历史 K/V。单 token decode 读取全部历史，prefill 使用因果掩码。达到学习式位置嵌入的上下文上限时，重新计算当前上下文上限内的最后一段 token，保持滑动窗口语义。测试覆盖缓存 logits 和上下文溢出后的贪心序列。

本机最终版本为 5,066,752 参数、256 token 上下文。artifacts/v2 包含推理权重、配置、词表、来源记录和 SHA256。HTTP 服务提供健康检查和生成接口，限制请求体与输出长度，同时只允许一次生成。忙时返回 429，参数错误返回 400。

性能报告记录固定提示、输出长度、线程数、缓存开关及重复测量中位数。HTTP 验收测 20 次串行请求的完整响应 P50/P95。它对应本机工作负载，持续可用性和外部生产负载需要独立验收。

## 8. 文件组织

~~~text
minigpt_v2/       PyTorch 模型、训练、数据处理、评测与服务
configs/          模型配置和实验计划
artifacts/        可直接加载的推理模型包
experiments/v2/   本轮训练日志、曲线、评测及选择记录
experiments/history/ 早期实验统计
data/             来源配方、词表与本地生成数据
tests/            正确性与 HTTP 回归测试
docs/             本说明、结果、模型卡说明及历史记录
~~~

训练 checkpoint 的优化器状态留在本机实验目录。仓库提供选定推理权重和重训命令。语料正文按配方重建。

## 9. 复现与演示

在仓库根目录创建并激活 Python 3.12 虚拟环境，再执行以下命令。Windows 使用 Scripts 目录中的 Python。

~~~bash
python -m pip install -r requirements-torch.txt
python -m unittest discover -s tests -v
python -m minigpt_v2.inference artifacts/v2 --prompt "The old man opened the door and" --count 64
python -m minigpt_v2.serve artifacts/v2
~~~

另一个终端调用服务：

~~~bash
curl http://127.0.0.1:8080/health
curl -H 'Content-Type: application/json' -d '{"prompt":"The old man opened the door and","max_new_tokens":32}' http://127.0.0.1:8080/generate
~~~

完整训练配方按顺序执行：

~~~bash
python prepare_corpus.py --download
python -m minigpt_v2.train --manifest data/corpus_manifest_27books.json --config configs/v2-large.json --output experiments/reproduce_large --steps 20000
python -m minigpt_v2.train --manifest data/corpus_manifest_27books.json --config configs/v2-large.json --init experiments/reproduce_large --output experiments/reproduce_refined --steps 5000 --learning-rate 0.0003 --min-learning-rate 0.00003
python -m minigpt_v2.normalize_corpus --manifest data/corpus_manifest_27books.json --control-manifest data/v2_raw_control/manifest.json
python -m minigpt_v2.train --manifest data/v2_normalized/manifest.json --config configs/v2-large.json --init experiments/reproduce_refined --output experiments/reproduce_normalized --steps 5000 --learning-rate 0.0003 --min-learning-rate 0.00003
python -m minigpt_v2.train --manifest data/v2_normalized/manifest.json --config configs/v2-context256.json --init experiments/reproduce_normalized --extend-context --validation-window 256 --output experiments/reproduce_context256 --steps 5000 --batch-size 4 --learning-rate 0.0003 --min-learning-rate 0.00003
python -m minigpt_v2.prepare_mixed
python -m minigpt_v2.train --manifest data/v2_mixed/manifest.json --config configs/v2-context256.json --init experiments/reproduce_context256 --validation-window 256 --output experiments/reproduce_mixed --steps 5000 --batch-size 4 --learning-rate 0.0003 --min-learning-rate 0.00003
python -m minigpt_v2.export experiments/reproduce_mixed artifacts/reproduced
~~~

复现格式对照时，将最后一次训练的 manifest 换成 data/v2_raw_control/manifest.json，并另设 output。

原计划中断后使用相同命令附加 --resume 对应目录/last.pt。新一阶段训练用 --init 开新目录。前者恢复优化器和采样状态，后者记录初始化权重指纹并重建状态。

最终评测与演示命令见 [结果页](results.md)。硬件、数据下载版本或库版本变化时，用指纹和日志确认比较条件。


## 10. 云端 98M 预训练与结项

沿用单卡 decoder 主线，将宽度扩展至 768、层数 12、头数 12、上下文 512，训练独立的 16,384 项 BPE。准备固定版本 FineWeb-Edu 分片，内容散列分离角色并精确去重，生成 uint16 内存映射文件，逐文件记录 SHA256。

先完成 46M 与 98M 两组顺序 100 步测速，测实际吞吐、显存和预算，再选择 98M 配置。保持训练协议与优化器状态，从第 100 步恢复至 122,071 步，累计约 20 亿目标。正式运行约 5 小时 7 分，最佳固定开发窗口 loss 3.10138，位于第 122,000 步。完整 LAMBADA 命中 896/5153，准确率 17.39%。最终权重另在完整验证和测试切分上得到 loss 3.03804 / PPL 20.8644 与 loss 3.03213 / PPL 20.7413；这些工程评测不能替代数据污染审查，也不能证明企业生产质量。

训练、导出、备份与公开评测约在 2026-11-21 04:49 完成，脚本关机遇到执行格式错误，预置平台定时关机在 07:30 兜底。实际总支出 15.27 元，包含约 5.04 元估算空闲费用。错误日志保留，调用兼容修复与回归检查另列。

云端全过程、真实命令、依赖及故障复盘见 [云端记录](CLOUD_TRAINING.md) 和 [结项记录](CLOSEOUT.md)。最终推理包通过版本附件分发，完整优化器状态保存在免费云端共享存储；位置与恢复边界见 [文件索引](FILE_INDEX.md)。
