# 🩺 Qwen2.5-0.5B 中文医疗问答微调 —— LoRA / DoRA / PiSSA 对比

在 Qwen2.5-0.5B-Instruct 上用中文医疗问答数据做 SFT，横向对比三种参数高效微调方法
（**LoRA**、**LoRA+DoRA**、**LoRA+PiSSA**）在 **held-out 测试集**（500 条，训练全程未使用）上的效果。

三种方法共享同一套数据、同一套超参、同一套评估流程，唯一变量是微调方法本身，
因此结论具备可比性。

## 📊 实验结果

在 500 条 held-out 测试集上评估。实验目录名由配置推导，格式为 `r{R}_alpha{ALPHA}_{挂载模块}`，
模块缩写按 q/k/v/o/gate/up/down 顺序（如 `qkvogud` = 全部 7 个投影层，`qv` = 仅 q_proj + v_proj）。

### 🥇 主实验：r=8, alpha=16, 7 个模块

| 方法 | 测试 Loss | Perplexity | 相对 Baseline | 训练 Loss | 泛化 gap | 训练耗时 | 平均回答长度 |
|------|----------:|-----------:|--------------:|----------:|---------:|---------:|-------------:|
| Baseline（原始模型） | 2.9455 | 19.02 | — | — | — | — | 313 字 |
| LoRA | 2.6065 | 13.55 | -11.5% | 2.5237 | 0.083 | 72 min | 173 字 |
| **LoRA+DoRA** | **2.6041** | **13.52** | **-11.6%** | 2.5160 | 0.088 | 150 min | 152 字 |
| LoRA+PiSSA | 2.6135 | 13.65 | -11.3% | 2.4629 | 0.151 | 67 min | 158 字 |

**几点观察：**

- ✅ 三种方法都把测试 loss 拉低了约 11.5%，微调确实有效。
- ⚠️ DoRA 测试 loss 最低，但训练耗时是 LoRA 的 **2 倍以上**（150 vs 72 min），收益仅 0.09%——性价比很低。
- ⚡ PiSSA 收敛最快（67 min）、训练 loss 最低，但泛化 gap 最大（0.151）。
- ✂️ 微调后回答长度从 313 字降到 ~160 字，模型学会了更精炼的回答风格。

### 🔬 配置对比：三组实验

| 实验配置 | 方法 | 测试 Loss | 相对 Baseline | 泛化 gap |
|---|---|---:|---:|---:|
| **r8 / alpha16 / 7 模块**（主实验） | LoRA | 2.6065 | -11.5% | 0.083 |
| | DoRA | 2.6041 | -11.6% | 0.088 |
| | PiSSA | 2.6135 | -11.3% | 0.151 |
| r16 / alpha32 / 7 模块 | LoRA | 2.5952 | -12.0% | 0.162 |
| | DoRA | 2.5920 | -12.1% | 0.167 |
| | PiSSA | 2.6120 | -11.4% | 0.272 |
| r8 / alpha16 / **2 模块**（仅 q,v） | LoRA | 2.6779 | -9.1% | **-0.023** |
| | DoRA | 2.6773 | -9.1% | -0.023 |
| | PiSSA | 2.6753 | -9.2% | -0.011 |

**结论：**

- 🎯 **挂载哪些模块比 r / alpha 重要得多。** r 从 16 降到 8、alpha 从 32 降到 16，测试 loss 只差
  约 0.5%（2.6041 → 2.5920），但 LoRA 可训练参数减少一半（rank 减半）——**故主实验选用
  r=8, alpha=16**。而只挂 q_proj/v_proj（丢掉 MLP 层）后掉到 **-9.1%**，差距约 2.5 个百分点。
- 📉 **2 模块组的 gap 是负的**（测试 loss 低于训练 loss），说明模型容量不足、处于欠拟合状态。
- 💡 结论：小模型上，MLP 层（gate/up/down）对容量贡献显著，**不建议只微调 attention 的 q/v**。

> 完整的逐题回答对比见 `eval_results/r8_alpha16_qkvogud/eval_results.json`。

## 📁 目录结构

带 `*` 的为占位文件（仅用于保留空目录，无实际内容）：

```
.
├── README.md                                  ← 本文件
├── requirements.txt                           依赖清单
├── test_qwen.py                               原始模型交互式对话（不套适配器），用于对照基线
│
├── scripts/                                   ★ 核心代码，4 个脚本按编号顺序执行
│   ├── prepare_data.py                        第 1 步：读 JSONL → 清洗 → Tokenize → Arrow
│   ├── preprocess.py                          第 2 步：PiSSA 前置，SVD 分解出残差模型
│   ├── train_lora.py                          第 3 步：训练，--method lora|dora|pissa
│   ├── evaluate.py                            第 4 步：对比评估（测试集 loss + 逐题回答）
│   └── peft/                                  上游 PEFT 官方示例，本项目代码的参考来源（见「致谢」）
│
├── eval_results/                              ★ 实验结果，按实验配置分目录
│   ├── r8_alpha16_qkvogud/                   主实验（r=8, alpha=16, 7 模块）
│   │   ├── summary.txt                       汇总表（测试 loss / PPL / gap / 耗时）
│   │   ├── eval_results.json                 逐题回答全文 + 各方法测试 loss
│   │   └── loss_compare.png                  训练 loss 曲线 + 测试 loss 对比柱状图
│   ├── r16_alpha32_qkvogud/                  同上（r=16, alpha=32, 7 模块）
│   │   ├── summary.txt
│   │   ├── eval_results.json
│   │   └── loss_compare.png
│   └── r8_alpha16_qv/                        同上（r=8, alpha=16, 仅 q/v 两个模块）
│       ├── summary.txt
│       ├── eval_results.json
│       └── loss_compare.png
│
├── qwen2.5_0.5B/.gitkeep *                    空占位：基座模型放这里（需自行下载，见下节）
├── data/.gitkeep *                            空占位：数据集放这里（需自行下载，见下节）
├── pissa_models/.gitkeep *                    空占位：由 preprocess.py 生成
└── adapter_save/.gitkeep *                    空占位：由 train_lora.py 生成
```

**实验目录命名规则**：`r{R}_alpha{ALPHA}_{模块缩写}`。模块缩写按
q / k / v / o / gate / up / down 顺序取首字母，例如：

- `r8_alpha16_qkvogud` —— r=8, alpha=16, 全部 7 个投影层（主实验）
- `r8_alpha16_qv` —— r=8, alpha=16, 仅 q_proj + v_proj（仅 attention）

训练、评估脚本都用同一套规则推导目录名，因此改 `--r/--alpha` 或模块列表后会自动落到新目录，
不会覆盖已跑过的实验。

> clone 后请按下一节「环境准备」把权重和数据放回对应目录。

## ⚙️ 环境准备

```bash
pip install -r requirements.txt
```

0.5B 模型很小，用 LoRA 微调的硬件要求很低：**单卡 8 GB 显存的消费级显卡即可**，
bf16 全精度训练即可，**无需 4bit 量化**；无 CUDA 的机器会自动降级为 fp32（很慢，仅用于跑通流程）。

显存更充足的话，可以关掉 [train_lora.py](scripts/train_lora.py) 里的 `gradient_checkpointing`
（它用计算换显存）来换取更快速度；`dataloader_num_workers` 按 CPU 核数调整。

### 1️⃣ 下载基座模型

放到项目根目录的 `qwen2.5_0.5B/`：

```bash
# 方式一：hf download
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir qwen2.5_0.5B

# 方式二：直接从 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct 下载全部文件放入该目录
```

最终目录里应有 `config.json`、`model.safetensors`、`tokenizer.json` 等文件。

### 2️⃣ 准备数据

原始数据为 JSONL，每行一条，字段为 `instruction` / `input` / `output`：

```json
{
  "instruction": "乙肝大三阳抗病毒治疗需要多久",
  "input": "",
  "output": "大三阳指乙肝大三阳，是对乙肝表面抗原阳性，乙肝e抗原阳性，乙肝核心抗体阳性的俗称……"
}
```

数据来源是 [shibing624/medical](https://huggingface.co/datasets/shibing624/medical) 的 `finetune` 子集，
**需自行下载**，然后放到 `data/raw_data/`：

```bash
# 方式一：hf download（只下 finetune 子集）
hf download shibing624/medical --repo-type dataset \
    --include "finetune/*" --local-dir ./hf_medical_tmp

# 方式二：浏览器打开
#   https://huggingface.co/datasets/shibing624/medical/tree/main/finetune
#   手动下载 train_zh_0.json / valid_zh_0.json / test_zh_0.json
```

把下载到的文件按下面的名字放到 `data/raw_data/`（**文件名须完全一致**，脚本按固定路径读取）：

```
data/raw_data/train_zh_0.json    # 训练集
data/raw_data/valid_zh_0.json    # 验证集
data/raw_data/test_zh_0.json     # 测试集
```

三份数据的分工（**术语在本文档中统一如下**）：

| 数据 | 条数 | 用途 | 谁读它 |
|------|-----:|------|--------|
| **训练集**（train） | 195 万（脚本采样 3 万） | 梯度更新 | `train_lora.py` |
| **验证集**（valid） | 500 | 训练中监控 `eval_loss`、挑最优 checkpoint | `train_lora.py` |
| **测试集**（test） | 500 | 最终评估，训练全程不使用 | `evaluate.py` |

> 验证集参与"挑哪个 checkpoint"这个决策，因此用它报成绩会偏乐观；测试集只在最后评估时读取一次，
> 从未参与任何训练决策，所以它是 **held-out 测试集**——本文档中的成绩均以测试集为准。

> 脚本默认 `MAX_SAMPLES=30000` 随机采样以提高迭代速度，如需全量训练改 [scripts/prepare_data.py](scripts/prepare_data.py) 顶部该参数。

## 🚀 使用流程

共 4 步，按顺序执行。其中**第 2 步仅 PiSSA 需要**，跑 LoRA / DoRA 可跳过。

#### 第 1 步：数据预处理 📥

```bash
python scripts/prepare_data.py          # → data/processed_data/
```

#### 第 2 步：PiSSA 分解（仅 PiSSA 需要）✂️

```bash
python scripts/preprocess.py            # → pissa_models/r8_alpha16/
```

**必须在跑 `--method pissa` 之前完成**。它把基座模型 SVD 分解成「残差模型 + pissa_init」，
训练时直接加载，不再现算 SVD。跑 LoRA / DoRA 不需要这一步。

> 第 1 步和第 2 步**互不依赖**（一个产出数据、一个产出残差模型），先后顺序随意。

#### 第 3 步：三种方法各训练一次 🏋️

```bash
python scripts/train_lora.py --method lora     # → adapter_save/r8_alpha16_qkvogud/adapter_lora/
python scripts/train_lora.py --method dora     # → .../adapter_dora/
python scripts/train_lora.py --method pissa    # → .../adapter_pissa/  （需先完成第 2 步）
```

#### 第 4 步：对比评估 📈

```bash
python scripts/evaluate.py              # → eval_results/r8_alpha16_qkvogud/
```

必须在三次训练都完成后执行。默认超参 `r=8, alpha=16`、7 个投影层，与主实验一致。

### 🔧 换其它配置

```bash
# r/alpha 可命令行指定（下例改用主实验对比过的 r16/alpha32）
# 模块列表改 train_lora.py 顶部的 LORA_TARGET_MODULES
python scripts/train_lora.py --method lora --r 16 --alpha 32

# 换配置后评估脚本顶部的 LORA_R / LORA_ALPHA 要同步修改，否则找不到 adapter 目录
# PiSSA 换配置还要重新分解（r/alpha 由 preprocess.py 决定）
python scripts/preprocess.py --lora_r 16 --lora_alpha 32
```

> **几点注意：**
> - `--r/--alpha` 只对 `--method lora/dora` 生效。**PiSSA 的 r/alpha 由 `preprocess.py` 决定**，
>   写死在生成好的 `pissa_init/adapter_config.json` 里，训练时传 `--r/--alpha` 会被忽略。
> - 评估脚本 [scripts/evaluate.py](scripts/evaluate.py) 顶部的 `LORA_R / LORA_ALPHA /
>   LORA_TARGET_MODULES` 必须与训练时一致，脚本据此推导 `adapter_save/` 下的实验目录。
> - 实验结果按实验配置分目录存放，改配置重跑不会覆盖已有结果。

### 💬 与原始模型对话

```bash
python test_qwen.py      # 交互式，输入 quit 退出，clear 清空历史
```

## 📦 下载适配器

适配器单个 14~48 MB（其中 `tokenizer.json` 固定占 11 MB，权重随 rank 变化），9 个合计约 270 MB。
因体积较大未入库，已发布到 HuggingFace Hub：

**https://huggingface.co/Chen-nwafu/qwen2.5-0.5b-medical-lora**

仓库内结构与本项目 `adapter_save/` 一一对应（3 个实验配置 × 3 种方法）：

```
r8_alpha16_qkvogud/     ← 主实验
├── adapter_lora/
├── adapter_dora/
└── adapter_pissa/
r16_alpha32_qkvogud/
└── ...
r8_alpha16_qv/
└── ...
```

下载后目录对上，可直接运行 `python scripts/evaluate.py`：

```
adapter_save/r8_alpha16_qkvogud/adapter_lora/     ← 评估脚本读这里
adapter_save/r8_alpha16_qkvogud/adapter_dora/
adapter_save/r8_alpha16_qkvogud/adapter_pissa/
...
```

## 🙏 致谢

**参考项目：** [MiniLoRA](https://github.com/SoloCalm/MiniLoRA) —— 本项目 SFT 流程的数据处理与
训练组织方式参考了该项目。

**数据集：** [shibing624/medical](https://huggingface.co/datasets/shibing624/medical)（`finetune` 子集）。

`scripts/peft/` 下的 DoRA 与 PiSSA 示例脚本来自 HuggingFace PEFT 官方仓库
（Apache-2.0），按其原始协议保留：

- [PEFT](https://github.com/huggingface/peft)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://huggingface.co/papers/2402.09353)
- [PiSSA: Principal Singular values and Singular vectors Adaptation](https://huggingface.co/papers/2404.02948)
- 基座模型：[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)（Apache-2.0）

## 📄 License

本项目代码未附协议文件，默认保留所有权利（All rights reserved）。

第三方组件与数据各有其原始协议：
- `scripts/peft/` 下的示例脚本：Apache-2.0（来源见上方致谢）
- 基座模型 Qwen2.5-0.5B-Instruct：Apache-2.0
- 训练数据集 shibing624/medical：Apache-2.0（数据集页面标注）
