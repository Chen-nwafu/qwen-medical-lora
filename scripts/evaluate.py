# -*- coding: utf-8 -*-
"""
evaluate.py — 第 4 步：对比评估
================================
加载三个训练好的适配器 + 原始模型（基线），从两个维度对比：

1. 测试集 Loss（定量） — 在 held-out 测试集上算 loss + perplexity
2. 问答质量（定性）     — 从测试集随机抽题，用两种解码方式各测一次

与其它步骤的关系：
    第 1 步 prepare_data.py   —— 数据准备
    第 2 步 preprocess.py     —— 仅 PiSSA 需要
    第 3 步 train_lora.py     —— 训练 lora / dora / pissa 三种
    第 4 步 evaluate.py       —— 本文件，必须在三次训练都完成后跑

运行方式：
    python scripts/evaluate.py

输出（按实验配置分目录，不会覆盖其它配置的结果）：
    eval_results/r{R}_alpha{ALPHA}_{模块}/  — loss_compare.png / eval_results.json / summary.txt

注意：脚本顶部的 LORA_R / LORA_ALPHA / LORA_TARGET_MODULES 必须与训练时一致，
否则会找不到 adapter_save/ 下对应的实验目录。
"""
import os
import json
import re
import math
import random
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
from torch.utils.data import DataLoader

# ============================================================
# 参数
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # scripts → 项目根目录
MODEL_PATH = os.path.join(PROJECT_ROOT, "qwen2.5_0.5B")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed_data", "test")
ORIGINAL_TEST_FILE = os.path.join(PROJECT_ROOT, "data", "raw_data", "test_zh_0.json")
MAX_NEW_TOKENS = 200
TEST_BATCH_SIZE = 4
NUM_SAMPLE_QUESTIONS = 5           # 从测试集随机抽几道做定性展示
SAMPLE_SEED = 123                  # 抽题用的种子（保证每次抽的一样）

# 须与训练时一致（train_lora.py 的 --r/--alpha、preprocess.py 的 --lora_r/--lora_alpha）
LORA_R, LORA_ALPHA = 8, 16
# 须与训练时挂的模块一致；目录名和 train_lora.py 用同一套推导规则
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
TARGET_ABBR = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
               "gate_proj": "g", "up_proj": "u", "down_proj": "d"}

EXPERIMENT = f"r{LORA_R}_alpha{LORA_ALPHA}_" + "".join(
    TARGET_ABBR[m] for m in LORA_TARGET_MODULES
)
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "adapter_save", EXPERIMENT)
PISSA_RESIDUAL_PATH = os.path.join(PROJECT_ROOT, "pissa_models", f"r{LORA_R}_alpha{LORA_ALPHA}")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "eval_results", EXPERIMENT)  # 按实验分目录，不覆盖其它配置

# (底模路径, adapter 路径)：PiSSA 用残差模型当底模，其余用原版底模
ADAPTERS = {
    "Baseline":       (MODEL_PATH, None),                       # ★ 基线：不用适配器
    "LoRA":           (MODEL_PATH, os.path.join(EXPERIMENT_DIR, "adapter_lora")),
    "LoRA+DoRA":      (MODEL_PATH, os.path.join(EXPERIMENT_DIR, "adapter_dora")),
    "LoRA+PiSSA":     (PISSA_RESIDUAL_PATH, os.path.join(EXPERIMENT_DIR, "adapter_pissa")),
}

SYSTEM_PROMPT = (
    "你是一位专业、耐心、严谨的中文医疗问答助手。"
    "回答要清晰、分点、通俗易懂，并提醒用户："
    "如果症状严重请及时就医，回答仅供参考，不能替代专业医疗建议。"
)


# ============================================================
# 1. 测试集 Loss（定量）
# ============================================================
def evaluate_test_loss(model, tokenizer) -> dict:
    if not os.path.exists(TEST_DATA_DIR):
        print(f"  [跳过] 找不到测试集: {TEST_DATA_DIR}")
        return {}
    ds = load_from_disk(TEST_DATA_DIR)
    print(f"  测试集 {len(ds)} 条")

    def collate_fn(batch):
        input_ids = [torch.tensor(x["input_ids"]) for x in batch]
        labels = [torch.tensor(x["labels"]) for x in batch]
        padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        # 生成 attention_mask，让模型忽略 padding 位置
        attention_mask = (padded != tokenizer.pad_token_id).long()
        return (
            padded[:, :1024].to(model.device),
            padded_labels[:, :1024].to(model.device),
            attention_mask[:, :1024].to(model.device),
        )

    loader = DataLoader(ds, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for input_ids, labels, attention_mask in loader:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return {"loss": avg_loss, "perplexity": ppl, "tokens": total_tokens}


# ============================================================
# 2. 从测试集加载原始问题（定性展示用）
# ============================================================
def load_sample_questions():
    """从原始测试集 JSONL 文件中随机抽取问题，返回 list[str]"""
    if not os.path.exists(ORIGINAL_TEST_FILE):
        return None
    questions = []
    with open(ORIGINAL_TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                q = item.get("instruction", "").strip()
                if q:
                    questions.append(q)
            except json.JSONDecodeError:
                continue
    if len(questions) > NUM_SAMPLE_QUESTIONS:
        random.seed(SAMPLE_SEED)
        questions = random.sample(questions, NUM_SAMPLE_QUESTIONS)
    return questions


# ============================================================
# 3. 生成回答（两种模式：可复现的 greedy + 多样性的 sample）
# ============================================================
def generate_answer(model, tokenizer, question, do_sample=False):
    """生成回答。
    do_sample=False: 贪婪解码，确定性输出，可复现
    do_sample=True:  随机采样，多样性高，展示创造性
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": do_sample,
        "repetition_penalty": 1.1,
    }
    if do_sample:
        gen_kwargs.update({"temperature": 0.7, "top_p": 0.9})
    # greedy 模式不需要 temperature/top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# 4. 画图
# ============================================================
def plot_results(test_losses: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 左图：训练 loss 曲线（仅三个适配器有，Baseline 没有）
    for name, (base_path, adapter_path) in ADAPTERS.items():
        if adapter_path is None:
            continue
        log_path = os.path.join(adapter_path, "loss_log.json")
        if not os.path.exists(log_path):
            continue
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        train = data.get("train", [])
        eval_log = data.get("eval", [])
        if train:
            ax1.plot([t["epoch"] for t in train], [t["loss"] for t in train],
                     label=f"{name} (train)")
        if eval_log:
            ax1.plot([e["epoch"] for e in eval_log], [e["eval_loss"] for e in eval_log],
                     linestyle="--", label=f"{name} (eval)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # 右图：所有（含 Baseline）测试集 Loss
    if test_losses:
        names = list(test_losses.keys())
        losses = [test_losses[n]["loss"] for n in names]
        colors = ["#A0A0A0" if "Baseline" in n else "#5B9BD5" for n in names]
        bars = ax2.bar(names, losses, color=colors)
        ax2.set_ylabel("Test Loss")
        ax2.set_title("Test Set Loss ↓ (含 Baseline)")
        ax2.tick_params(axis="x", rotation=15)
        for bar, loss in zip(bars, losses):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{loss:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, "loss_compare.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n对比图已保存 → {save_path}")


# ============================================================
# 5. 统计
# ============================================================
def compute_stats(text):
    text = text.strip()
    sentences = len(re.split(r"[。！？\n]", text)) if text else 0
    return {"chars": len(text), "sentences": max(1, sentences)}


# ============================================================
# 6. 读取 adapter 补充信息（训练配置 + train loss + 耗时）
# ============================================================
def load_adapter_meta(adapter_path) -> dict:
    """从 adapter 目录的 adapter_config.json / loss_log.json 读取补充信息，
    用于汇总表展示实验配置、train loss、gap、耗时。"""
    meta = {}
    if not adapter_path or not os.path.isdir(adapter_path):
        return meta
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(cfg_path):
        try:
            cfg = json.load(open(cfg_path, encoding="utf-8"))
            meta["r"] = cfg.get("r")
            meta["alpha"] = cfg.get("lora_alpha")
            meta["dropout"] = cfg.get("lora_dropout")
            meta["n_modules"] = len(cfg.get("target_modules", []))
        except (json.JSONDecodeError, OSError):
            pass
    log_path = os.path.join(adapter_path, "loss_log.json")
    if os.path.exists(log_path):
        try:
            lg = json.load(open(log_path, encoding="utf-8"))
            trains = lg.get("train", [])
            if trains:
                meta["train_loss"] = trains[-1]["loss"]
            meta["elapsed_s"] = lg.get("elapsed_s")
        except (json.JSONDecodeError, OSError):
            pass
    return meta


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  适配器效果对比评估（含 Baseline）")
    print("=" * 60)

    # 只加载 tokenizer（各模型的底模在循环里按需加载，保证每轮都是干净的模型）
    print(f"\n[1] 加载 tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 从测试集抽题
    sample_questions = load_sample_questions()
    if sample_questions:
        print(f"\n从测试集抽取 {len(sample_questions)} 道题:")
        # for i, q in enumerate(sample_questions, 1):
        #     print(f"  {i}. {q[:60]}...")
    else:
        print("\n[警告] 找不到原始测试集，跳过定性评估")

    all_answers = {}
    test_losses = {}

    # 逐个评估（含 Baseline），每轮从自己的底模路径重新加载
    for name, (base_path, adapter_path) in ADAPTERS.items():
        print(f"\n{'=' * 50}")
        print(f"  评估: {name}")
        print(f"{'=' * 50}")

        # 每轮独立加载底模（bf16，与训练时一致；PiSSA 用残差模型）
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto", trust_remote_code=True,
        )
        if adapter_path is not None:
            if not os.path.exists(adapter_path):
                print(f"  [跳过] 找不到适配器: {adapter_path}")
                del model
                continue
            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()

        # --- 定量：测试集 loss ---
        print(f"\n  [{name}] 测试集 Loss ...")
        tl = evaluate_test_loss(model, tokenizer)
        if tl:
            test_losses[name] = tl
            print(f"    Loss: {tl['loss']:.4f}  |  Perplexity: {tl['perplexity']:.2f}")

        # --- 定性：两种解码模式 ---
        if sample_questions:
            answers = []
            for q in sample_questions:
                # greedy: 可复现，用于客观对比
                ans_greedy = generate_answer(model, tokenizer, q, do_sample=False)
                # sample: 展示多样性
                ans_sample = generate_answer(model, tokenizer, q, do_sample=True)
                stats = compute_stats(ans_greedy)
                answers.append({
                    "question": q,
                    "greedy": ans_greedy,
                    "sample": ans_sample,
                    **stats,
                })
                # print(f"    Q: {q[:50]}...")
                # print(f"    [greedy] {ans_greedy[:100]}...")
                # print(f"    ({stats['chars']} 字)\n")
            all_answers[name] = answers

        # 每轮都卸载，避免显存累积（各方法底模独立加载，互不影响）
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 收集各 adapter 的补充信息（train loss / 耗时 / 配置）
    adapter_meta = {}
    for name, (base_path, adapter_path) in ADAPTERS.items():
        adapter_meta[name] = load_adapter_meta(adapter_path)

    # 画图
    print("\n[2] 生成对比图 ...")
    plot_results(test_losses)

    # 保存
    results = {"test_losses": test_losses, "answers": all_answers}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 汇总表（同时打印 + 保存为 txt）
    summary_lines = []
    summary_lines.append("=" * 72)
    summary_lines.append("  评估汇总")
    summary_lines.append("=" * 72)

    # A. 实验身份信息（从 adapter config 读取）
    first_adapter = next((p for _, (_, p) in ADAPTERS.items() if p), None)
    exp_cfg = load_adapter_meta(first_adapter) if first_adapter else {}
    if exp_cfg:
        summary_lines.append(
            f"  实验配置: r={exp_cfg.get('r')}  alpha={exp_cfg.get('alpha')}  "
            f"target模块={exp_cfg.get('n_modules')}个"
        )
    summary_lines.append(f"  评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("=" * 72)

    # B. 汇总表（含 train loss / gap / 耗时）
    summary_lines.append(f"  {'方法':<16} {'测试Loss':>8} {'PPL':>7} {'Train':>8} {'gap':>7} {'Δ%':>7} {'耗时':>6} {'长度':>5}")
    summary_lines.append("  " + "-" * 70)

    baseline_loss = test_losses.get("Baseline", {}).get("loss", None)
    for name in all_answers:
        tl = test_losses.get(name, {})
        loss = tl.get("loss", None)
        loss_str = f"{loss:.4f}" if loss else "N/A"
        ppl_str = f"{tl.get('perplexity', float('nan')):.2f}" if tl else "N/A"
        # train loss / gap（测试loss - train末值，正值越大说明train远低于test，过拟合倾向越强）/ 耗时
        meta = adapter_meta.get(name, {})
        train_loss = meta.get("train_loss")
        if train_loss is not None and loss:
            train_str = f"{train_loss:.4f}"
            gap_str = f"{loss - train_loss:.3f}"
        else:
            train_str = "—"
            gap_str = "—"
        dur_str = f"{meta['elapsed_s']/60:.0f}m" if meta.get("elapsed_s") else "—"
        # 相对于 Baseline 的改善
        if loss and baseline_loss and "Baseline" not in name:
            delta = (baseline_loss - loss) / baseline_loss * 100
            delta_str = f"-{delta:.1f}%" if delta > 0 else f"+{-delta:.1f}%"
        elif "Baseline" in name:
            delta_str = "—"
        else:
            delta_str = "N/A"
        avg_len = sum(a["chars"] for a in all_answers[name]) / len(all_answers[name])
        summary_lines.append(
            f"  {name:<16} {loss_str:>8} {ppl_str:>7} {train_str:>8} {gap_str:>7} {delta_str:>7} {dur_str:>6} {avg_len:>4.0f} 字"
        )

    if test_losses:
        # 去掉 baseline 再比最优
        finetuned = {k: v for k, v in test_losses.items() if "Baseline" not in k}
        if finetuned:
            best = min(finetuned, key=lambda k: finetuned[k]["loss"])
            summary_lines.append(f"\n  ★ 微调方法中 Loss 最优: {best}")
        if baseline_loss:
            summary_lines.append(f"  ★ Baseline Loss: {baseline_loss:.4f}（微调后应低于此值）")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # 保存汇总表到 txt
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n汇总表已保存 → {summary_path}")

    print("\n全部完成！")


if __name__ == "__main__":
    main()
