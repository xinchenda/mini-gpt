# 文件索引与保留范围

项目主目录为 `mini-gpt/`，Git 工作区已归到该目录。训练、推理和测试均从主目录运行。

| 路径 | 用途 | 保存方式 |
| --- | --- | --- |
| README.md | 项目入口与复现命令 | Git |
| MODEL_CARD.md | 云端 98M 模型的任务、数据、指标和限制 | Git |
| docs/local-v2-model-card.md | 本机 5M 模型的历史说明 | Git |
| docs/CLOSEOUT.md | 最终结果、费用和故障复盘 | Git |
| docs/TRAINING_GUIDE.md | 完整训练阶段与历史演进 | Git |
| docs/CLOUD_TRAINING.md | 云平台、数据上传、GPU 环境、费用与训练流程 | Git |
| docs/results.md | 本机实测结果和失败范围 | Git |
| minigpt_v2/ | 模型、训练、数据处理、评测、推理与服务 | Git |
| configs/ | 候选模型配置与实验计划 | Git |
| scripts/ | 云端预算守护和自动测速 | Git |
| tests/ | 恢复、因果性、缓存、数据、预算与接口回归检查 | Git |
| artifacts/cloud/ | 云端最佳权重推理包，下载后校验 | GitHub Release 附件，普通 Git 忽略 |
| cloud_training_state/（AutoDL 共享盘） | 完整模型、优化器、随机数和采样器状态 | 云端保留，用于精确恢复，未声称已下载本机 |
| artifacts/v2/ | 本机最终混合语料推理模型包 | Git |
| artifacts/v2_literary/ | 本机纯文学上下文 256 对照模型包 | Git |
| artifacts/v2_context64/、v2_raw_format/ | 历史格式与上下文对照权重 | 本机保留，Git 忽略 |
| experiments/v2/ | 11 组本机实验的日志、曲线、报告与完整训练状态 | 报告入 Git，优化器 checkpoint 本机保留 |
| experiments/cloud/ | 云端数据审计、环境、预算、测速与训练证据 | 报告入 Git，训练状态按体积单独保存 |
| experiments/history/ | 精选早期学习实验统计 | Git |
| experiments/ 其他历史目录 | 字符、MLX、上下文与多种子实验 | 本机完整保留，发布按记录说明 |
| data/ | 语料来源配方、manifest、词表、审计和本机正文 | 配方与审计入 Git，正文和二进制 token 不上传 Git |
| data/web_cloud/ | FineWeb-Edu 原始分片、编码 token 和校验记录 | 本机保留；云端传输包仅含编码结果 |
| exports/ | 原 v1 ZIP、云端代码与数据传输包 | 本机保留，Git 忽略 |
| archive/pre_v2_sources/ | 更新前被替换文件的逐文件备份 | 本机保留，Git 忽略 |
| archive/file_inventory_before_v2.json | 整理前 382 个文件的大小与 SHA256 | 本机保留 |
| docs/file-audit.json | 整理后逐文件核对结果 | Git |
| train.py、train_mlx.py、train_text_mlx.py 等 | 前期学习与 MLX v1 复现入口 | 保留兼容路径，主线从 minigpt_v2 进入 |

整理先备份再替换，没有删除旧训练权重、旧数据或旧代码。初始 382 个文件均在主目录或替换前备份中找到对应 SHA256，未发现遗漏。缓存、虚拟环境、构建目录和归档本身不纳入这份初始清单。

旧构建目录已在浏览器上传完成后归入 `archive/build_workspace_2026-11-20/`；其中保留构建代码和传输 ZIP，它不再是 Git 工作区。历史日志中的原执行绝对路径保留，避免改写训练来源记录。

最终云端模型超过 Git 普通单文件大小限制时，权重应作为版本附件发布；仓库保留精确版本、SHA256 和下载命令。正式训练、评测和备份完成前不发布最终版本。
