"""
python test.py
"""

import json
from swebench.inference.make_datasets.utils import extract_diff 


def find_golden_patch(instance_id):
    data_path = "/home/featurize/data/AI-ModelScope/SWE-bench/data/test-00000-of-00001.json"

    # 1. 打开文件并加载为 Python 对象
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 自动把 JSON 转成 Python dict 或 list
    
    for item in data:
        if item.get("instance_id") == instance_id:
            return item.get("patch")
    return None


def find_model_patch(instance_id):
    """
    获得模型输出的patch
    """
    data_path = "/home/featurize/opencompass/outputs/default/20260105_170717/predictions/qwen2.5-coder-1.5b-instruct-vllm/swebench-bm25-27k.json"
    # 1. 打开文件并加载为 Python 对象
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 自动把 JSON 转成 Python dict 或 list
    data = list(data.values())  # 转成 list 方便遍历
    for item in data:
        if item.get("gold") == instance_id:
            return item.get("prediction")
    return None


def find_model_prompt(instance_id):
    """
    获得模型输出的patch
    """
    data_path = "/home/featurize/opencompass/outputs/default/20260105_170717/predictions/qwen2.5-coder-1.5b-instruct-vllm/swebench-bm25-27k.json"
    # 1. 打开文件并加载为 Python 对象
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 自动把 JSON 转成 Python dict 或 list
    data = list(data.values())  # 转成 list 方便遍历
    for item in data:
        if item.get("gold") == instance_id:
            return item.get("origin_prompt")
    return None


if __name__ == "__main__":
    instance_id = "astropy__astropy-11693"
    patch = find_golden_patch(instance_id)
    print(f"Golden patch for instance {instance_id}: \n {patch}")

    print('---------------')

    pre_patch = find_model_patch(instance_id)
    print(f"predict patch for instance {instance_id}: \n {pre_patch}")

    print('---------------')
    diff = extract_diff(pre_patch)
    print(f"extract diff {instance_id}: \n {diff}")

    
    # print('---------------')
    # prompt = find_model_prompt(instance_id)
    # print(f"predict prompt for instance {instance_id}: \n {prompt}")