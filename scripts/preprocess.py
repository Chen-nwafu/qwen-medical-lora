# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
preprocess.py —— 第 2 步（仅 PiSSA 需要）：SVD 分解出残差模型
==============================================================
把基座模型按 SVD 拆成「残差模型 + pissa_init 初始 adapter」，供 PiSSA 训练使用。

为什么需要它：PiSSA 训练时要读预分解结果。若跳过这步，train_lora.py 找不到
pissa_models/r{R}_alpha{ALPHA}/ 目录会直接报错。

只需跑一次，之后 train_lora.py --method pissa 直接加载，不再现算 SVD。

与其它步骤的关系：
    第 1 步 prepare_data.py   —— 数据准备（与本步互不依赖，先后任意）
    第 2 步 preprocess.py     —— 本文件，仅 PiSSA 需要
    第 3 步 train_lora.py     —— 训练，必须在本步之后
    第 4 步 evaluate.py       —— 评估

运行方式：
    python scripts/preprocess.py                              # 默认 r=16 alpha=32
    python scripts/preprocess.py --lora_r 8 --lora_alpha 16   # 换配置需重新分解

输出：
    pissa_models/r{R}_alpha{ALPHA}/               — 残差模型
    pissa_models/r{R}_alpha{ALPHA}/pissa_init/    — 初始 adapter

注意：r/alpha 必须与训练脚本和评估脚本一致，否则路径对不上。
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # scripts → 项目根目录

parser = argparse.ArgumentParser(description="PiSSA 分解：把基座模型拆成残差模型 + pissa_init adapter")
parser.add_argument(
    "--base_model_name_or_path",
    default=os.path.join(PROJECT_ROOT, "qwen2.5_0.5B"),
    help="The name or path of the fp32/16 base model. (默认: 项目内 qwen2.5_0.5B)",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="输出目录；默认自动生成 pissa_models/r{rank}_alpha{alpha}",
)
parser.add_argument("--bits", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
parser.add_argument(
    "--init_lora_weights", type=str, default="pissa", help="(`['pissa', 'pissa_niter_[number of iters]']`)"
)
parser.add_argument("--lora_r", type=int, default=8)     # 与 train_lora.py 的 LORA_R 保持一致
parser.add_argument("--lora_alpha", type=int, default=16)  # 与 train_lora.py 的 LORA_ALPHA 保持一致
parser.add_argument("--lora_dropout", type=int, default=0)
script_args = parser.parse_args()
if script_args.output_dir is None:
    # 默认输出路径与 train_lora.py / evaluate.py 里的 PISSA_RESIDUAL_PATH 完全一致
    script_args.output_dir = os.path.join(
        PROJECT_ROOT, "pissa_models", f"r{script_args.lora_r}_alpha{script_args.lora_alpha}"
    )
print(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.base_model_name_or_path,
    dtype=(
        torch.float16
        if script_args.bits == "fp16"
        else (torch.bfloat16 if script_args.bits == "bf16" else torch.float32)
    ),
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    script_args.base_model_name_or_path, trust_remote_code=True
)
tokenizer.pad_token_id = tokenizer.eos_token_id
lora_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    init_lora_weights=script_args.init_lora_weights,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, lora_config)

# Save PiSSA modules:
peft_model.peft_config["default"].init_lora_weights = True
peft_model.save_pretrained(os.path.join(script_args.output_dir, "pissa_init"))
# Save residual model:
peft_model = peft_model.unload()
peft_model.save_pretrained(script_args.output_dir)
# Save the tokenizer:
tokenizer.save_pretrained(script_args.output_dir)
