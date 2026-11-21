# MiniGPT 开源方案对照审查

审查日期：2026-11-21。范围为当前单卡英文语言模型预训练，预算仍为 50 元。审查采用官方源码与项目记录，未因本次审查中断云端训练、改变其配方或启动额外收费实验。

## 结论

基础预训练路线成立，适合作为小模型训练工程项目：因果 decoder、下一 token 目标、AdamW、BF16、梯度裁剪、验证选模、完整状态恢复和数据指纹均有实现。但尚未证明是高质量通用模型，也未通过企业部署验收。优化重点应转向数据覆盖和可复核评测，再通过对照验证训练配方细节。

本次审查补上此前缺少的系统开源对照。发现差异不等于现有运行无效；参考方案也不能脱离数据、批次和硬件直接照搬。

## 参考选择

| 项目 | 本次读取的内容 | 许可证与维护判断 | 采用方式 |
| --- | --- | --- | --- |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | model.py、train.py、README、LICENSE | MIT；README 已注明旧实现被弃用并推荐 nanochat | 作为简洁 GPT-2 风格结构与优化器基线，不视为最新框架 |
| [nanochat](https://github.com/karpathy/nanochat) | README、LICENSE | MIT；覆盖从词表到预训练、微调、评测和推理的路线 | 后续端到端流程参考；不照搬其多卡硬件和费用宣传 |
| [LitGPT](https://github.com/Lightning-AI/litgpt) | pretrain.py、仓库说明、LICENSE.md | Apache-2.0；有配置化预训练与恢复入口 | 借鉴配置、验证、checkpoint 和训练预算组织方式；暂不引入完整 Lightning 栈 |
| [SmolLM](https://github.com/huggingface/smollm) | text/pretraining/README.md、LICENSE | Apache-2.0；当前主线覆盖多个模型世代 | 借鉴数据、阶段与消融组织；不照搬大规模分布式实现 |

读取时的 HEAD：nanoGPT `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`；nanochat `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`；LitGPT `5341568a65e918a85cb284fac038de2157eebf3f`；SmolLM `438be18dae658a638f3a6976fdf5190bb6d68619`。固定版本的读取结果和 SHA256 见 [来源审计](../experiments/reviews/opensource_sources.json)。维护判断只适用于本次读取，未来使用时重新核验。

本次没有复制第三方源码，也没有新增这些框架的依赖。以后实际复用时，保留对应许可和版权信息，单独检查模型及数据许可。

## 当前实现与差异

| 环节 | 本项目事实 | 对照判断与处理 |
| --- | --- | --- |
| 结构 | Pre-LN、因果多头 SDPA、GELU、共享嵌入 | 与 nanoGPT 风格基线一致；可以保留 |
| 初始化 | 线性层统一标准差 0.02 | nanoGPT 对残差投影另做深度缩放。本项目未采用；后续只在新初始化对照中验证 |
| 优化器 | 所有参数统一 weight decay 0.1；AdamW 默认 betas | nanoGPT 按参数维度分组，偏置/归一化不衰减，并显式设置 betas。当前配方需披露，后续公平对照后再选 |
| 学习率 | 最高 0.0006、余弦到 0.00006、固定 warmup 100 步 | 调度形式合理；预热与有效 token 数尚未系统验证，不能直接借用其他批次的步数 |
| 数据 | 训练侧拟合 BPE，内容散列划分，精确去重，EOS packing，内存映射 | 基础分离成立；缺少近似去重和公开任务污染检查 |
| 重复训练 | 约 3 亿准备 token，约 20 亿累计训练目标 | 相当于约 6.67 倍随机曝光；不能当成 20 亿不同 token，也不能据此保证计算最优 |
| 文档边界 | 云端 packing 允许跨 EOS 注意力 | 与 SmolLM3 的文档内 masking 存在差异；这是可比较的设计选择，不据此认定错误 |
| 开发集 | 固定 64 个 512-token 窗口，共 32,768 个评分目标 | 可监控和选模，但并非完整约 147 万 token 的验证集；结项补充全覆盖评测，单独记录评分位置 |
| 恢复 | 模型、优化器、采样器、CPU/CUDA RNG 与协议 | CPU 连续/中断恢复有逐参数一致性测试；CUDA 正常恢复已执行，尚未单独证明 GPU 位级一致性 |
| 存盘 | 每 100 步检查并写 last，改善时写 best | 大模型优化器状态带来较多 I/O。后续将日志、验证和存盘频率独立配置并测速 |
| 验收 | 自实现 LAMBADA runner；云端已完成 5,153 题，17.39%；另保留脚本关机失败 | 增加独立标准实现交叉核验和固定生成样例；不能仅看 loss，也不能用未完成的评测宣称成功 |
| 工程边界 | 单卡任务、限时保存、平台定时关机、本机接口 | 适合预算内可复现实验；不等于企业分布式训练或生产 SLA |

源码依据：

- [nanoGPT 结构、初始化和参数分组](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py)
- [nanoGPT 训练与学习率配置](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py)
- [LitGPT 预训练和验证](https://github.com/Lightning-AI/litgpt/blob/5341568a65e918a85cb284fac038de2157eebf3f/litgpt/pretrain.py)
- [SmolLM 预训练与文档边界](https://github.com/huggingface/smollm/blob/438be18dae658a638f3a6976fdf5190bb6d68619/text/pretraining/README.md)

## 下一阶段的验收顺序

1. 核验本轮是否完成、最佳/最终步数、训练曲线、数据和权重指纹、实际费用及关机状态。未完成或限时退出需明确标记。
2. 固定最佳权重，补充覆盖更广的验证与独立测试，保存生成失败样例，核查公共任务实现及可能的数据污染。不同词表的 token PPL 不直接比较。
3. 若重复训练出现收益停滞，先检验更丰富且经过质量处理的数据。若仍持续改善，再评估续训的边际收益和费用。
4. 用短程、同数据、同初始化或同随机种子、同训练目标预算的实验，分别检查优化器分组、betas、预热和存盘频率。残差初始化只能用从头训练对照，不能改写已有权重后冒充同一实验。
5. 根据目标任务决定指令微调或视觉扩展。多模态先检索成熟连接层与视觉编码器方案，建立任务和评测，再决定是否训练；本次没有新增多模态预算。

## 持续改进记录要求

后续重要变更必须记录：待解决问题、官方参考及版本、复用或自行实现的取舍、改动范围、控制变量、训练/评测费用、结果及是否保留。已有运行保留为基线，新增方案分目录记录。提升必须由实验支持，不以参数增大、功能数量或开源项目 star 数替代效果。
