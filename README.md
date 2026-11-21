# MiniGPT：单卡语言模型预训练与评测

从字符级学习实验发展到约 98M 参数的英文 GPT 类语言模型，记录数据处理、受控对照、可恢复训练、公开任务评测、模型导出和云端费用。本轮云端预训练已完成，实付 15.28 元。

| 实测项目 | 结果 |
| --- | --- |
| 参数量 | 98,032,128，12 层、宽度 768、12 头 |
| 上下文 / 词表 | 512 token / 16,384 Byte-Level BPE |
| 训练数据 / 累计目标 | 约 3 亿准备 token / 约 20 亿累计训练目标 |
| 完成训练 | 122,071 步，单张 RTX 4090D，BF16 |
| 最佳固定开发窗口 loss | 3.10138，第 122,000 步 |
| 完整验证切分 | loss 3.03804，PPL 20.8644，1,467,199 个目标 |
| 完整独立测试切分 | loss 3.03213，PPL 20.7413，1,499,973 个目标 |
| LAMBADA OpenAI | 896/5153，准确率 17.39%，最后词 PPL 206.79 |

开发窗口分数不是完整验证集分数。公开任务污染尚未审查，LAMBADA 也曾用于前期诊断。模型用于英文续写与训练工程复现，输出仍会出错，未完成指令微调，不能视为通用助手或企业生产模型。

## 阅读入口

| 文档 | 内容 |
| --- | --- |
| [结项记录](docs/CLOSEOUT.md) | 实际结果、费用、关机故障与恢复 |
| [完整训练记录](docs/TRAINING_GUIDE.md) | 从梯度练习、MLX 到 PyTorch 和云端的阶段演进 |
| [云端操作记录](docs/CLOUD_TRAINING.md) | 平台选择、数据传输、测速、预算与训练命令 |
| [模型卡](MODEL_CARD.md) | 模型、数据、评测及适用范围 |
| [开源方案审查](docs/OPEN_SOURCE_REVIEW.md) | 官方实现对照、现有差异与改进优先级 |
| [文件索引](docs/FILE_INDEX.md) | 发布内容、训练状态与历史文件的位置 |
| [本机 v2 结果](docs/results.md) | 约 507 万参数的前序模型及对照 |
| [MLX v1](docs/v1-release.md) | 最初 27 本书训练记录 |

## 运行环境与验证

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-torch.txt
.venv/bin/python -m unittest discover -s tests -v
```

Windows 使用 `.venv/Scripts/python.exe`。CUDA 训练实测环境为 Python 3.12.3、PyTorch 2.8.0+cu128、RTX 4090D。详细依赖见 [云端环境](experiments/cloud/cloud_requirements.txt)；Mac 实测依赖另外保留。

## 获取云端模型

大权重通过本仓库版本附件分发，代码仓库保留配置、数据配方和实验记录。私有仓库需要先登录 GitHub CLI：

```bash
gh release download v2.0.0 --repo xinchenda/mini-gpt --pattern minigpt-cloud98m.zip --dir exports
python -m zipfile -e exports/minigpt-cloud98m.zip .
python -m minigpt_v2.inference artifacts/cloud --prompt "The old man opened the door and" --count 64
python -m minigpt_v2.serve artifacts/cloud
```

推理包校验权重、词表、配置及来源记录的 SHA256。完整优化器状态另行保存，不混入推理包。旧本机模型包 `artifacts/v2/` 与 `artifacts/v2_literary/` 保留，便于复核演进。

## 训练复现

```bash
python -m pip install -r requirements-cloud.txt
python -m minigpt_v2.prepare_web --output data/web_cloud --max-train-tokens 300000000
python -m minigpt_v2.train \
  --manifest data/web_cloud/manifest.json --config configs/v2-cloud98m.json \
  --output experiments/my_cloud --steps 122071 --batch-size 32 \
  --device cuda --precision bf16 --learning-rate 0.0006 --min-learning-rate 0.00006
```

实际运行先完成 100 步 pilot，再用相同协议恢复。`--resume` 恢复模型、优化器、采样器和 RNG，要求数据指纹与训练协议一致；`--init` 用于另开训练阶段。云服务器必须另设平台关机时限，训练程序退出不代表停止计费。

## 工程与结果边界

测试覆盖因果性、KV-cache、目标评分覆盖、训练恢复、数据分离、模型包完整性、HTTP 边界与预算/关机处理。CPU 的连续与中断恢复通过逐参数一致性测试，未声称 GPU 位级一致性。HTTP 默认仅监听本机，不提供生产 SLA。

项目使用 Codex 辅助实现与验证。开源资料的对照、未采用的设计和真实故障均保留记录；未将成熟设计或一次实验包装为原创算法。数据正文按官方来源配方重建，不提交原始网页或书籍正文。
