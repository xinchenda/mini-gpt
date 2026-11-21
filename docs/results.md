# 本机 v2 结果与验收

本页对应约 507 万参数的本机模型包 `artifacts/v2`。后续云端候选单独记录，未完成时不替换本页指标。

## 语言建模

| 文本 | MLX v1 token PPL | PyTorch v2 token PPL | v2 bits/byte |
| --- | ---: | ---: | ---: |
| The Lost World 留出书 | 921.00 | 23.53 | 1.7358 |
| WikiText-2 测试切分 | 2706.47 | 10.29 | 1.6140 |

证据：[留出书完整报告](../experiments/v2/final_evaluation.json)、[WikiText-2 完整报告](../experiments/v2/wikitext2.json)。两版使用同一 1,024 项 BPE、相同文本、步幅 16，并覆盖每个下一 token 目标一次。v1 使用原生上下文 64，v2 使用原生上下文 256。模型结构、数据格式与训练量均改变，结果是完整配方比较。

v1 对连续段落有明显排版失配；不能把这组大幅下降全部归因于新增参数。清洗对照与共同验证集记录见实验目录。WikiText-2 使用 PyTorch 官方仓库的标准 tokenized 文本，未采用 GPT-2 论文的去 tokenization 协议。其 token PPL 10.29 不代表达到论文中 word PPL 相近数字的能力；本项目同次测得的空白词 PPL 为 339.51。

## 公开任务暴露的能力缺口

完整 LAMBADA OpenAI 共 5,153 题，v2 最后一个词的完整贪心预测命中 **5 题，准确率 0.097%**，目标词 PPL 约 846,535。v1 为 0 题。公开任务曾用于发现硬换行偏置，属于测试知情开发，不能作为完全盲测成绩。

[完整任务结果](../experiments/v2/lambada_full.json)保留数据指纹和评分协议。GPT-2 论文 117M 模型的 LAMBADA 准确率为 45.99%，其训练数据量、词表、上下文和输出约束与本项目不同，仅作历史能力参考。[原论文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

当前模型可生成部分局部连贯的英语句子，仍有重复、语义跳跃和长程依赖失败。项目证明的是可复现训练与评测流程，不支持把它部署为通用问答或决策模型。

## 推理与接口

在 MacBook Air M4 上以 CPU 两线程、8 token 固定提示、24 token 输出测量，预热后各运行 5 次取中位数：

| 模式 | 生成 token/s |
| --- | ---: |
| 不使用 KV-cache | 863.34 |
| 使用 KV-cache | 1689.00 |

两种模式贪心输出完全一致。数字对应短输出、无上下文溢出的条件；[长输出报告](../experiments/v2/benchmark_cpu_long.json)另行保留。它不是并发服务吞吐，也不是云 GPU 性能。

[HTTP 本机验收](../experiments/v2/http_acceptance.json)对 32 token 输出进行了 20 次串行请求，20 次成功、0 次失败，完整响应 P95 为 19.35 ms。项目内部门槛为 P95 小于 2 秒，已通过。这一小样本验收不代表生产 SLA。

## 可复核证据

- [完整训练阶段与预算](../experiments/v2/run_catalog.json)
- [固定提示的生成样例](../experiments/v2/generation_samples.json)
- [容量验证曲线](../experiments/v2/validation_curves.png)
- [训练全流程](TRAINING_GUIDE.md)
- [公开指标口径](industry-metrics.md)
- [云端训练记录](CLOUD_TRAINING.md)

运行 `python -m unittest discover -s tests -v` 检查因果性、缓存、数据分离、连续与恢复训练、模型包校验及接口边界。结项回归测试为 22 项通过；CUDA 路径的实际训练结果见 [云端结项](CLOSEOUT.md)。
