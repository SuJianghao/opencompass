#!/usr/bin/env python3

import json
from swebench.harness.test_spec.test_spec import MAP_REPO_VERSION_TO_SPECS

# ===== 配置 =====
DATA_PATH = "/home/featurize/data/AI-ModelScope/SWE-bench/data/test-00000-of-00001.json"  # 你的SWE-bench测试集文件（或转换后的JSON文件）
# =================

# 读取数据
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

invalid_samples = []

for sample in data:
    repo = sample.get("repo")
    version = sample.get("version")
    
    try:
        if repo not in MAP_REPO_VERSION_TO_SPECS:
            invalid_samples.append(sample)
        elif version not in MAP_REPO_VERSION_TO_SPECS[repo]:
            invalid_samples.append(sample)
    except Exception as e:
        # 理论上不会进来这里，除非结构有问题
        print(f"[异常] repo={repo}, version={version}, error={e}")
        invalid_samples.append(sample)

# 打印结果
print(f"总样本数: {len(data)}")
print(f"无对应测试规范的样本数: {len(invalid_samples)}")
print("-" * 50)
for s in invalid_samples:
    print(json.dumps(s, ensure_ascii=False))

# 如果你只想看 repo 和 version：
print("\n出问题的 repo-version 对：")
for s in invalid_samples:
    print(f"{s.get('repo')}  {s.get('version')}")
