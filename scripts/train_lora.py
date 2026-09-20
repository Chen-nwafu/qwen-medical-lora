# -*- coding: utf-8 -*-
"""
train_lora.py — 第 3 步：LoRA / DoRA / PiSSA 训练
==================================================
加载预处理好的数据，用 bf16 全精度对 Qwen2.5-0.5B-Instruct 做 SFT。

与其它步骤的关系：
    第 1 步 prepare_data.py   —— 数据准备，必须先跑（本脚本读 data/processed_data/）
    第 2 步 preprocess.py     —— 仅 PiSSA 需要，跑 --method pissa 前必须先跑
    第 3 步 train_lora.py     —— 本文件，三种方法各跑一次
    第 4 步 evaluate.py       —— 评估，必须在三次训练都完成后跑

运行方式：
    python scripts/train_lora.py --method lora
    python scripts/train_lora.py --method dora
    python scripts/train_lora.py --method pissa

输出：adapter_save/<实验配置>/adapter_<method>/
"""
import os
import time
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, PeftModel, get_peft_model
from datasets import load_from_disk

# ============================================================
# ★ 可调参数
# ============================================================

# ---- 微调方法：命令行指定，例：python scripts/train_lora.py --method dora ----
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--method", choices=["lora", "dora", "pissa"], default="lora")
# 注意：--r/--alpha 只对 lora / dora 生效。PiSSA 的 r/alpha 由 preprocess.py 决定，
# 固定写在生成好的 pissa_init adapter_config.json 里，这里传了也不会被读取。
_parser.add_argument("--r", type=int, default=8, help="LoRA rank（默认 8，与主实验结果一致；PiSSA 无效）")
_parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha（默认 16，与主实验结果一致；PiSSA 无效）")
_args = _parser.parse_args()
METHOD = _args.method

# ---- 训练超参 ----
BATCH_SIZE = 16
GRAD_ACCUM = 2           # 等效 batch = 16 * 2 = 32
NUM_EPOCHS = 3
SEED = 42

# ---- LoRA 参数 ----
LORA_R = _args.r
LORA_ALPHA = _args.alpha
# ---- 各方法的学习率 / dropout（dropout 按方法单独设置）----
METHOD_HYPER = {
    "lora":  {"learning_rate": 2e-4, "lora_dropout": 0.05},
    "dora":  {"learning_rate": 2e-4, "lora_dropout": 0.05},  # DoRA 收敛慢，调优时可试 lr=3e-4 或 r=8
    "pissa": {"learning_rate": 2e-4, "lora_dropout": 0.0},   # PiSSA 官方 dropout=0，与 LoRA/DoRA 不同
}
LEARNING_RATE = METHOD_HYPER[METHOD]["learning_rate"]
LORA_DROPOUT = METHOD_HYPER[METHOD]["lora_dropout"]
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
TARGET_ABBR = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
               "gate_proj": "g", "up_proj": "u", "down_proj": "d"}

# ---- 路径 ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # scripts → 项目根目录
MODEL_PATH = os.path.join(PROJECT_ROOT, "qwen2.5_0.5B")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed_data")
# 实验目录名由配置推导（与 evaluate.py、preprocess.py 保持一致）：
#   r16_alpha32_qkvogud / r8_alpha16_qv ...
EXPERIMENT = f"r{LORA_R}_alpha{LORA_ALPHA}_" + "".join(
    TARGET_ABBR[m] for m in LORA_TARGET_MODULES
)
ADAPTER_DIR = os.path.join(PROJECT_ROOT, "adapter_save", EXPERIMENT, f"adapter_{METHOD}")
# PiSSA 残差模型路径（由 preprocess.py 生成，r/alpha 变了要重新分解）
PISSA_RESIDUAL_PATH = os.path.join(PROJECT_ROOT, "pissa_models", f"r{LORA_R}_alpha{LORA_ALPHA}")

# ============================================================
# 根据 METHOD 生成 LoRA 额外参数
# ============================================================
METHOD_CONFIG = {
    "lora":  {},
    "dora":   {"use_dora": True},
    "pissa":  {"init_lora_weights": "pissa"},
}
if METHOD not in METHOD_CONFIG:
    raise ValueError(f"未知的 METHOD: {METHOD}，可选: {list(METHOD_CONFIG.keys())}")
lora_extra = METHOD_CONFIG[METHOD]

print("=" * 60)
print(f"  LoRA 微调训练 — 方法: {METHOD}")
print(f"  模型: {MODEL_PATH}")
print(f"  适配器输出: {ADAPTER_DIR}")
print("=" * 60)

# ============================================================
# 加载预处理好的数据
# ============================================================
print("\n[1/5] 加载预处理数据 ...")
train_ds = load_from_disk(os.path.join(DATA_DIR, "train"))
eval_ds = load_from_disk(os.path.join(DATA_DIR, "eval"))
print(f"  训练集 {len(train_ds)} 条 / 验证集 {len(eval_ds)} 条")

# ============================================================
# 加载 Tokenizer
# ============================================================
print(f"\n[2/5] 加载 tokenizer: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True, use_fast=False
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 加载 bf16 全精度模型 + 套 LoRA
# ============================================================
print(f"\n[3/5] 加载 bf16 模型 + LoRA (method={METHOD}) ...")

# PiSSA 用预分解的残差模型（pissa_models/，SVD 只跑过一次），其余方法用原版底模
base_path = PISSA_RESIDUAL_PATH if METHOD == "pissa" else MODEL_PATH
model = AutoModelForCausalLM.from_pretrained(
    base_path,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False       # 训练时关闭 KV 缓存，省显存（推理不需要）
model.enable_input_require_grads()   # 让输入参与梯度计算，LoRA 层才能反向传播（PEFT 推荐）

if METHOD == "pissa":
    # 加载残差模型 + pissa_init 初始 adapter，只训练 A/B，不再现算 SVD
    model = PeftModel.from_pretrained(
        model, PISSA_RESIDUAL_PATH, subfolder="pissa_init", is_trainable=True
    )
else:
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        **lora_extra,
    )
    model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# Trainer 配置
# ============================================================
print(f"\n[4/5] 配置 Trainer ...")

training_args = TrainingArguments(
    output_dir=os.path.join(PROJECT_ROOT, "tmp", "checkpoints", METHOD),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=1,              # 只保留最优的一个 checkpoint，省空间
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,     # ★ 训练结束后加载最优权重
    report_to=[],
    seed=SEED,
    dataloader_num_workers=16,       # 数据加载并行进程数，按机器核数调整（核多可调大加速）
    dataloader_pin_memory=True,      # 加速 CPU→GPU 数据拷贝
    gradient_checkpointing=True,     # 用计算换显存：激活不全存，反向时重算（省约 70% 激活显存）
    bf16=torch.cuda.is_available(),  
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
)

# ============================================================
# 训练
# ============================================================
print(f"\n[5/5] 开始训练 ...")
t0 = time.time()
trainer.train()
elapsed = time.time() - t0

# ---- 保存适配器 ----
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"\n适配器已保存 → {ADAPTER_DIR}")
print(f"训练耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")

# ---- 收集 loss 并单独保存（供后续评估使用） ----
losses = {
    "method": METHOD,
    "train": [
        {"epoch": log["epoch"], "loss": log["loss"]}
        for log in trainer.state.log_history if "loss" in log
    ],
    "eval": [
        {"epoch": log["epoch"], "eval_loss": log["eval_loss"]}
        for log in trainer.state.log_history if "eval_loss" in log
    ],
    "elapsed_s": elapsed,
}
loss_path = os.path.join(ADAPTER_DIR, "loss_log.json")
with open(loss_path, "w", encoding="utf-8") as f:
    json.dump(losses, f, ensure_ascii=False, indent=2)
print(f"Loss 日志已保存 → {loss_path}")

# ---- 画单组 loss 曲线 ----
plt.figure(figsize=(6, 4))
train_logs = [l for l in trainer.state.log_history if "loss" in l]
eval_logs = [l for l in trainer.state.log_history if "eval_loss" in l]
plt.plot([l["epoch"] for l in train_logs], [l["loss"] for l in train_logs], label="train")
if eval_logs:
    plt.plot([l["epoch"] for l in eval_logs], [l["eval_loss"] for l in eval_logs],
             label="eval", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Loss — {METHOD}")
plt.legend()
plt.grid(alpha=0.3)
fig_path = os.path.join(ADAPTER_DIR, "loss.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Loss 曲线已保存 → {fig_path}")
print("\n完成！")
