#!/usr/bin/env python3
# python count_long_prompts.py
import json
from transformers import AutoTokenizer

# ===== 写死的参数 =====
MODEL_PATH = "/home/featurize/base_model/Qwen/Qwen2.5-Coder-1.5B-Instruct"   # 你的模型目录路径
JSON_FILE  = "/home/featurize/work/princeton-nlp/SWE-bench_bm25_13K/data/test-00000-of-00001.json"  # 要统计的JSON文件路径
MAX_TOKENS = 32768                   # 长度阈值
IF_TIKTOKEN = False                      # True = 用 tiktoken(cl100k_base)，False = 用 HF tokenizer
# =====================

if IF_TIKTOKEN:
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")

    def get_token_length(prompt: str) -> int:
        """用 tiktoken 计算 cl100k_base token 数"""
        return len(encoding.encode(prompt))

else:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def get_token_length(prompt: str) -> int:
        """用 HuggingFace tokenizer 计算 token 数"""
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        return len(token_ids)


def main():
    # 读取 JSON 文件
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_count = 0
    over_count = 0

    for obj in data:
        if "text" not in obj:
            continue  # 跳过没有 text 的对象
        total_count += 1
        length = get_token_length(obj["text"])
        if length > MAX_TOKENS:
            over_count += 1

    if total_count == 0:
        print("没有找到有效的 'text' 字段")
        return

    ratio = over_count / total_count * 100
    print(f"总样本数: {total_count}")
    print(f"超过 {MAX_TOKENS} tokens 的样本数: {over_count}")
    print(f"占比: {ratio:.2f}%")


if __name__ == "__main__":
    main()