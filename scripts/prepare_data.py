"""
prepare_data.py — 第 1 步：数据预处理
======================================
读取本地已下载的 JSONL 数据，
格式化为 Chat Template，Tokenize，保存到本地。

只需跑一次，后续训练脚本直接加载处理好的数据。

与其它步骤的关系：
    第 1 步 prepare_data.py   —— 本文件
    第 2 步 preprocess.py     —— 仅 PiSSA 需要（与本步互不依赖，先后任意）
    第 3 步 train_lora.py     —— 训练
    第 4 步 evaluate.py       —— 评估

运行方式：
    python scripts/prepare_data.py

输入：
    data/raw_data/train_zh_0.json    — 训练集 (JSONL)
    data/raw_data/valid_zh_0.json    — 验证集 (JSONL，训练中监控用)
    data/raw_data/test_zh_0.json     — 测试集 (JSONL，最终评估用)

输出：
    data/processed_data/train/       — 训练集 (Arrow)
    data/processed_data/eval/        — 验证集
    data/processed_data/test/        — 测试集（训练全程不动，只用于最终评估）
"""
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from datasets import Dataset

# ============================================================
# 可调参数
# ============================================================
MAX_SAMPLES = 30000                           # 最多用多少条训练数据（总数据 195 万条，全量训练太慢，故随机采样）
MAX_LEN = 512                                 # 最大 token 长度
IGNORE_INDEX = -100                           # CrossEntropyLoss 的 ignore_index，用于屏蔽 prompt 部分
SEED = 42

# 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # scripts → 项目根目录
MODEL_PATH = os.path.join(PROJECT_ROOT, "qwen2.5_0.5B")
TRAIN_FILE = os.path.join(PROJECT_ROOT, "data", "raw_data", "train_zh_0.json")
VALID_FILE = os.path.join(PROJECT_ROOT, "data", "raw_data", "valid_zh_0.json")
TEST_FILE = os.path.join(PROJECT_ROOT, "data", "raw_data", "test_zh_0.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed_data")

SYSTEM_PROMPT = (
    "你是一位专业、耐心、严谨的中文医疗问答助手。"
    "回答要清晰、分点、通俗易懂，并提醒用户："
    "如果症状严重请及时就医，回答仅供参考，不能替代专业医疗建议。"
)


# ============================================================
# 工具函数：读取 JSONL
# ============================================================
def load_jsonl(path):
    """逐行读取 JSONL 文件，返回 list[dict]，跳过格式损坏的行"""
    data = []
    bad_lines = 0
    print(f"  读取: {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
                continue
    if bad_lines:
        print(f"  跳过 {bad_lines} 条格式损坏的行")
    print(f"  共 {len(data)} 条")
    return data


# ============================================================
# 工具函数：清洗数据
# ============================================================
def clean_items(raw_items: list[dict]) -> list[dict]:
    """清洗原始数据，返回统一格式的 list[dict]

    处理的脏数据情况：
      - output 为空              → 丢弃（没有回答，无法做 SFT）
      - instruction 为空，input 有值 → 用 input 作为问题
      - 两者都有                 → 拼接作为问题
      - 两者都为空               → 丢弃（没有问题，无法训练）

    返回统一格式：
      {"question": "完整问题", "output": "回答"}
    """
    cleaned = []
    stats = {"empty_output": 0, "empty_question": 0}

    for item in raw_items:
        instruction = str(item.get("instruction") or "").strip()
        input_text = str(item.get("input") or "").strip()
        output = str(item.get("output") or "").strip()

        # 过滤 1: output 为空 → 丢弃
        if not output:
            stats["empty_output"] += 1
            continue

        # 构造问题：合并 instruction + input
        if instruction and input_text:
            question = f"{instruction}\n{input_text}"
        elif instruction:
            question = instruction
        elif input_text:
            question = input_text
        else:
            stats["empty_question"] += 1
            continue

        cleaned.append({"question": question, "output": output})

    if cleaned:
        print(f"  清洗后保留 {len(cleaned)} 条")
        print(f"  过滤: output为空 {stats['empty_output']} 条, 问题为空 {stats['empty_question']} 条")
    return cleaned


# ============================================================
# 第 1 步：加载本地数据
# ============================================================
print(f"[1/4] 加载本地数据 ...")

raw_train = load_jsonl(TRAIN_FILE)
raw_valid = load_jsonl(VALID_FILE)
raw_test = load_jsonl(TEST_FILE)

# 训练集截取（195 万条太多，取子集）
if MAX_SAMPLES and len(raw_train) > MAX_SAMPLES:
    import random
    random.seed(SEED)
    raw_train = random.sample(raw_train, MAX_SAMPLES)
    print(f"  随机采样 {MAX_SAMPLES} 条")

# 清洗数据
print(f"\n  清洗训练集 ...")
clean_train = clean_items(raw_train)
print(f"  清洗验证集 ...")
clean_valid = clean_items(raw_valid)
print(f"  清洗测试集 ...")
clean_test = clean_items(raw_test)

# 看一眼清洗后的格式
sample = clean_train[0]
print(f"\n  清洗后字段: {list(sample.keys())}")
# print(f"  样例 question: {sample['question'][:80]}")
# print(f"  样例 output  : {sample['output'][:80]}")

train_ds = Dataset.from_list(clean_train)
eval_ds = Dataset.from_list(clean_valid)
test_ds = Dataset.from_list(clean_test)
print(f"\n  训练集 {len(train_ds)} 条 / 验证集 {len(eval_ds)} 条 / 测试集 {len(test_ds)} 条")

# ============================================================
# 第 2 步：加载 Tokenizer
# ============================================================
print(f"\n[2/4] 加载 tokenizer: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True, use_fast=False
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 第 3 步：格式化 + Tokenize
# ============================================================
print("\n[3/4] 格式化 + Tokenize ...")


def format_chat(example):
    """格式化完整对话（system + user + assistant），生成 text 列（仅用于展示）"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def tokenize_fn(example):
    """Tokenize + 构造 labels（assistant-only loss mask，参考 MiniLoRA 的做法）

    labels 中 prompt 部分（system + user）设为 -100，不参与 loss 计算；
    只有 assistant 回答部分保留真实 token id，参与 loss。这是 SFT 的标准做法：
    模型只学习"怎么回答"，不需要学习"复述问题"。
    """
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True, return_tensors=None
    )
    # 注意：apply_chat_template 的返回类型因版本而异（list / BatchEncoding / tensor）
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()

    answer_ids = tokenizer(
        example["output"] + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids

    # 超过最大长度时从开头截断（保留 prompt，截掉过长的回答）
    if len(input_ids) > MAX_LEN:
        input_ids = input_ids[:MAX_LEN]
        labels = labels[:MAX_LEN]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


# 第一步：格式化（生成 text 列）
train_ds = train_ds.map(format_chat)
eval_ds = eval_ds.map(format_chat)
test_ds = test_ds.map(format_chat)

# print(f"  训练样本 (前 200 字符):\n    {train_ds[0]['text'][:200]}...")

# 第二步：Tokenize，并删除所有原始列（question/output/text）
train_ds = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(tokenize_fn, remove_columns=eval_ds.column_names)
test_ds = test_ds.map(tokenize_fn, remove_columns=test_ds.column_names)

# ============================================================
# 第 4 步：保存
# ============================================================
print(f"\n[4/4] 保存 → {OUTPUT_DIR}")
os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "eval"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

train_ds.save_to_disk(os.path.join(OUTPUT_DIR, "train"))
eval_ds.save_to_disk(os.path.join(OUTPUT_DIR, "eval"))
test_ds.save_to_disk(os.path.join(OUTPUT_DIR, "test"))

print(f"  完成！训练集 {len(train_ds)} 条 / 验证集 {len(eval_ds)} 条 / 测试集 {len(test_ds)} 条")
# print(f"  接下来运行: python scripts/train_lora.py")