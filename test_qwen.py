"""
Qwen2.5-0.5B-Instruct 原始模型对话测试脚本
用途：加载本地 Qwen2.5-0.5B 模型（未套适配器），进行交互式对话测试
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== 配置 =====================
MODEL_PATH = "qwen2.5_0.5B"                    # 本地模型路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20

# ===================== 加载模型 =====================
print(f"[INFO] 设备: {DEVICE} | 精度: {TORCH_DTYPE}")
print(f"[INFO] 正在加载模型: {MODEL_PATH} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # 生成时需要一个 pad_token，缺失则补上
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=TORCH_DTYPE,
    device_map="auto" if DEVICE == "cuda" else "cpu",
    trust_remote_code=True,
)

model.eval()
print(f"[INFO] 模型加载完成！参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# ===================== 对话函数 =====================
def chat(prompt: str, history: list | None = None) -> str:
    """
    单轮对话（支持历史记录）
    Args:
        prompt: 用户输入
        history: [(user_msg, assistant_msg), ...]
    Returns:
        模型回复
    """
    if history is None:
        history = []

    # Qwen2.5 使用 chat_template，直接 apply 即可
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码（只取生成部分，去掉输入）
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()

# ===================== 交互式对话 =====================
def interactive_chat():
    """交互式多轮对话"""
    print("\n" + "=" * 50)
    print("  Qwen2.5 交互式对话（输入 'quit' 退出，'clear' 清空历史）")
    print("=" * 50 + "\n")

    history = []

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] 对话结束")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("[INFO] 对话结束")
            break

        if user_input.lower() == "clear":
            history.clear()
            print("[INFO] 历史记录已清空\n")
            continue

        print("\n🤖 Qwen2 思考中...", end="\r")
        response = chat(user_input, history)
        print(f"🤖 Qwen2: {response}\n")

        history.append((user_input, response))



if __name__ == "__main__":
    interactive_chat()
