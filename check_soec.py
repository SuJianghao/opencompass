# python check_soec.py
import json
from swebench.harness.test_spec.test_spec import MAP_REPO_VERSION_TO_SPECS

# 你的本地数据文件路径
DATA_PATH = "/home/featurize/work/princeton-nlp/SWE-bench_bm25_13K/data/test-00000-of-00001.json"   # 如果是 .jsonl 就改成对应路径

# ===== 加载 JSON 数据 =====
if DATA_PATH.endswith(".json"):
    # JSON 数组格式
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
elif DATA_PATH.endswith(".jsonl"):
    # JSON Lines 格式
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
else:
    raise ValueError("请提供 .json 或 .jsonl 文件路径")

# ===== 扫描不在 MAP_REPO_VERSION_TO_SPECS 的样本 =====
bad_samples = []
bad_version_5 = []

for sample in dataset:
    repo = sample.get("repo")
    version = str(sample.get("version"))  # 转成字符串避免 int/str 混用

    if repo not in MAP_REPO_VERSION_TO_SPECS:
        bad_samples.append({
            "instance_id": sample.get("instance_id"),
            "repo": repo,
            "version": version,
            "reason": "repo_not_found"
        })
        continue

    if version not in MAP_REPO_VERSION_TO_SPECS[repo]:
        reason = "version_not_found"
        if version == "5":
            bad_version_5.append({
                "instance_id": sample.get("instance_id"),
                "repo": repo,
                "version": version
            })
            reason = "version_is_5"
        bad_samples.append({
            "instance_id": sample.get("instance_id"),
            "repo": repo,
            "version": version,
            "reason": reason
        })

# ===== 输出结果 =====
print("=" * 80)
print(f"检测到 {len(bad_samples)} 条 repo+version 不在 MAP_REPO_VERSION_TO_SPECS 中")
for bad in bad_samples:
    print(f"{bad['instance_id']} | {bad['repo']} | {bad['version']} | {bad['reason']}")

print("=" * 80)
print(f"其中 version = '5' 的样本共有 {len(bad_version_5)} 条：")
for bad in bad_version_5:
    print(f"{bad['instance_id']} | {bad['repo']} | {bad['version']}")
