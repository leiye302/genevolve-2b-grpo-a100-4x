# GenEvolve 2B：SFT → GRPO only（4×A100 80GB）

这是 [GenEvolve-2B-Reproduction 的配对消融](https://github.com/leiye302/GenEvolve-2B-Reproduction/tree/67f58aaa6c4324739a5da6cd78b8a33021cb41aa)，不是作者发布的 8B 模型或训练入口。复用同一 **Qwen3-VL-2B-Instruct SFT 导出**，从二阶段开始，只做在线 GRPO。保留官方三种工具、静态知识技能、渲染器、图像与程序奖励、参考策略 KL、采样设置和评测。删除视觉经验提取、检索/记忆、特权 Teacher 和 SDL；不需要经验模型，也没有新增 RETHINK 或 OPSD。

[GenEvolve 论文](https://arxiv.org/html/2605.21605v2) · [作者工具和评分代码](https://github.com/MeiGen-AI/GenEvolve/tree/23c847c559ccc0f95bbf4b3d8925898463822f4c) · [固定版本](sources.json)

## 启动

在分配好的四卡 Linux x86-64 训练机上使用 Python 3.11、CUDA 12.8 toolkit、可用的 NVIDIA 驱动及 C/C++ 编译器。四张 A100 80GB 是与对照仓库一致的目标配置；实际显存峰值以目标机试跑为准。渲染模型缓存、图片和可续训断点都要留盘。保留对照仓库的 FSDP 和 CPU offload 设置。

### 1. 安装并指定现有 SFT

```bash
git clone https://github.com/leiye302/genevolve-2b-grpo-a100-4x.git
cd genevolve-2b-grpo-a100-4x
bash scripts/setup.sh
source .venv/bin/activate

export WORK=/path/to/new-grpo-run
export SFT_CHECKPOINT=/path/to/existing-sft-hf-export
python run.py check-sft --sft-checkpoint "$SFT_CHECKPOINT"
python run.py --work "$WORK" prepare
```

`SFT_CHECKPOINT` 指向对照实验也会使用的完整 HF 导出，例如原仓库的 `work/sft` 最终目录，或包含相同 HF 文件的 `checkpoint-550`。需要配置、分词器、处理器和全部 safetensors 权重分片。不需要 SFT 优化器；只有 ZeRO-3 原生分片时，请先用原 SFT 工程导出完整 HF 权重，再指向导出目录。`check-sft` 不调用 GPU 或 API，只做结构检查并输出文件 SHA256；可以和方法端输入逐项比对。不会回退到基模。

`prepare` 只下载固定版本的 GenEvolve RL 数据（2,575 训练、600 验证）和 594 题 GenEvolve-Bench，不下载 SFT 数据、基模或 embedding 模型。公开资源版本见 `sources.json`。限流时可自行设置 `HF_TOKEN`，不要提交任何凭据。

### 2. 配置渲染和评分

使用与对照相同的 Qwen-Image-Edit-2511 兼容服务（`POST /generate`）。如果没有现成服务，可在渲染机器运行：

```bash
bash scripts/setup.sh --renderer
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_renderer.sh --cpu-offload
```

本地渲染器使用固定版本的 Qwen-Image-Edit-2511；客户端沿用 40 步、CFG 4.0、seed 0、最多 2 张参考图。同机部署时渲染器会与训练共享显存和 CPU 内存，建议先确认峰值，或沿用已有渲染服务。评审仍是配对仓库的 `qwen3.7-flash-2026-07-15`，关闭思考。请在自己的 shell 中配置 Serper 和百炼 Key：

```bash
# 仅在未由调度器指定时，设置为自己实际获配的 4 张卡。
export CUDA_VISIBLE_DEVICES=0,1,2,3
export QWEN_SERVICE_URL=http://127.0.0.1:8001
export JUDGE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
read -rsp 'Serper API key: ' SERPER_API_KEY; printf '\n'
read -rsp 'Model Studio API key: ' JUDGE_API_KEY; printf '\n'
export SERPER_API_KEY JUDGE_API_KEY
```

搜索和奖励评分仍消耗 API 额度。这里取消的是经验总结调用，不是奖励 Judge。环境变量模板见 [configs/services.env.example](configs/services.env.example)。

### 3. 只跑二阶段 GRPO

```bash
python run.py --work "$WORK" grpo --sft-checkpoint "$SFT_CHECKPOINT"
```

默认与对照仓库一致：每次更新 8 个请求、每个 6 条轨迹（四卡各 2 个完整 group），遍历同一训练集一轮，共 322 次更新。`--steps N` 仅供配对短跑使用；短跑的输出请放在不同的工作目录。中断后用同一命令、同一数据、SFT 导出和总步数重启。续训检查 SFT 文件指纹、训练 JSONL 指纹、配置、Judge、世界大小及训练总步数。

| 路径（相对于 `WORK`） | 产物 |
| --- | --- |
| `grpo/contract.json` | 运行配置、SFT SHA256、训练清单 SHA256、公开源版本 |
| `grpo/rollouts/`、`grpo/metrics/` | 逐步轨迹、渲染图、原始奖励、有效程序比例、GRPO loss、学习率、梯度范数 |
| `grpo/checkpoint-*/` | 每 10 步及最后一步保存，保留最近两个完整状态，包含模型、优化器、调度器和 RNG |
| `grpo/final/` | 最终 HF safetensors 导出 |

## 对齐口径

- 生成：seed 42、temperature 0.7、top-p 0.95，最多 11 次模型调用、30,000 个响应 token。沿用原 agent prompt、`search`、`image_search`、`query_knowledge` 和最终 prompt-reference program。
- 奖励：同一有效程序判定，图像 KScore 和五档程序充分性各占 0.5；无效程序为 0。冻结的同一 SFT 权重只给原采样 token 算 reference log-prob，用于原 GRPO 奖励 KL 惩罚（系数 0.001）；**不是蒸馏 Teacher**。
- 优化：沿用原不对称裁剪（下 0.20、上 0.28）、按同题组归一化优势、轨迹内 token loss 求和再对轨迹平均、LR 1e-6 及 cosine 调度。相对配对仓库，训练目标只移除 SDL：`GRPO + 2 × SDL → GRPO`。这不是把经验权重置零后仍跑提取和 Teacher；这些路径不在本仓库中。

## 可选评测

训练不会自动发起付费评测。需要比较时，用两边相同的渲染、Judge、seed 和 split。默认全量 GenEvolve-Bench（594 题），`--split rl-val` 为固定的 600 条 RL 验证集。可用 `--policy` 对同一个 SFT 起点评测；小规模 `--max-cases N` 只用于试跑。

```bash
python run.py --work "$WORK" eval --split bench
# 可选：评测同一 SFT 起点
python run.py --work "$WORK" eval --split bench --policy "$SFT_CHECKPOINT"
```

保留对照仓库的独立 WISE legacy 评测（1,000 题）；仅需要它时运行 `bash scripts/setup.sh --wise-sources`，并配置提供 `gpt-4o-2024-05-13` 的 `WISE_API_KEY`（兼容端点可另设 `WISE_BASE_URL`），再执行 `python run.py --work "$WORK" eval --split wise`。

## 离线验收

```bash
GENEVOLVE_PYTHON=python3.11 bash scripts/setup.sh --sources-only
python3.11 -m venv .venv-smoke
source .venv-smoke/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-smoke.txt
python -m unittest discover -s tests -v
```

CPU 检查用小模型和模拟服务执行真实 rollout token 评分、奖励、反向传播、断点恢复、导出与评测汇总。Linux CI 另覆盖四 rank FSDP/Gloo；仅在纯 CPU 测试中跳过设备到 CPU 的 offload，CUDA 仍用对照仓库的配置。它们不代表目标机上的四卡 CUDA、对方实际 SFT 文件或真实外部服务已经验证。
