import json
import os
import re
from os import environ

from datasets import Features, Value, Dataset, DatasetDict
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from ..base import BaseDataset

from pathlib import Path
from typing import List, Any

from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS


from opencompass.datasets.SWE_bench.utils import eval_instance, find_golden_patch, check_data
from swebench.inference.make_datasets.utils import extract_diff 


@LOAD_DATASET.register_module()
class SWEBenchDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)

        def collect_files(prefix):
            """按文件名前缀收集 .json / .jsonl 文件"""
            return sorted([
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.startswith(prefix)
                and (f.endswith('.json') or f.endswith('.jsonl'))
            ])

        def load_json_file(file_path):
            if file_path.endswith(".json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_path.endswith(".jsonl"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return [json.loads(line) for line in f]
            else:
                raise ValueError("请提供 .json 或 .jsonl 文件路径")

        # 手动读取，确保 version 保持 str
        train_data = [item for f in collect_files('train') for item in load_json_file(f)]
        dev_data   = [item for f in collect_files('dev') for item in load_json_file(f)]
        test_data  = [item for f in collect_files('test') for item in load_json_file(f)]

        # 创建 DatasetDict
        dataset = DatasetDict({
            "train": Dataset.from_list(train_data),
            "dev": Dataset.from_list(dev_data),
            "test": Dataset.from_list(test_data)
        })

        # check_data(dataset['test'])
        return dataset

# @LOAD_DATASET.register_module()
# class SWEBenchDataset(BaseDataset):

#     @staticmethod
#     def load(path):
#         path = get_data_path(path)

#         def collect_files(prefix):
#             """按文件名前缀收集 .json / .jsonl 文件"""
#             return sorted([
#                 os.path.join(path, f)
#                 for f in os.listdir(path)
#                 if f.startswith(prefix)
#                 and (f.endswith('.json') or f.endswith('.jsonl'))
#             ])

#         train_files = collect_files('train')
#         dev_files = collect_files('dev')
#         test_files = collect_files('test')

#         # ==== 这里是关键改动 ====
#         features = Features({
#             "instance_id": Value("string"),
#             "text": Value("string"),
#             "repo": Value("string"),
#             "base_commit": Value("string"),
#             "problem_statement": Value("string"),
#             "hints_text": Value("string"),
#             "created_at": Value("string"),
#             "patch": Value("string"),
#             "test_patch": Value("string"),
#             "version": Value("string"),
#             "FAIL_TO_PASS": Value("string"),
#             "PASS_TO_PASS": Value("string"),
#             "environment_setup_commit": Value("string")
#         })

#         dataset = hf_datasets.load_dataset(
#             'json',
#             data_files={
#                 'train': train_files,
#                 'dev': dev_files,
#                 'test': test_files
#             },
#             features=features  # 👉 强制列类型
#         )
#         # ========================
#         check_data(dataset['test'])
#         return dataset


def compute_swebench_metrics(report: dict) -> dict:
    """
    根据 SWE-bench 的评测 report 计算四个指标：
    - %Resolved：FTP=1 & PTP=1 的比例
    - %Apply：patch 应用成功比例
    - avg_FTP_rate：平均 FAIL_TO_PASS 测试通过率
    - avg_PTP_rate：平均 PASS_TO_PASS 测试保持通过率

    Args:
        report (dict): {instance_id: {...}} 格式的评测结果

    Returns:
        dict: 包含上述四个指标的字典
    """
    total_count = 0
    resolved_count = 0
    apply_count = 0
    ftp_rates = []
    ptp_rates = []

    for instance_id, inst_report in report.items():
        total_count += 1

        # 如果是字符串，说明是Patch Apply Failed，直接跳过
        if "error" in inst_report:
            continue

        # 应用成功
        if inst_report.get(instance_id, {}).get('patch_successfully_applied', False):
            apply_count += 1

        # 提取测试状态并计算 FTP/PTP
        ftp_rate = None
        ptp_rate = None
        if 'tests_status' in inst_report.get(instance_id, {}):
            ts = inst_report.get(instance_id, {})['tests_status']
            # FAIL_TO_PASS
            ftp_success = len(ts['FAIL_TO_PASS']['success'])
            ftp_failure = len(ts['FAIL_TO_PASS']['failure'])
            if (ftp_success + ftp_failure) > 0:
                ftp_rate = ftp_success / (ftp_success + ftp_failure)

            # PASS_TO_PASS
            ptp_success = len(ts['PASS_TO_PASS']['success'])
            ptp_failure = len(ts['PASS_TO_PASS']['failure'])
            if (ptp_success + ptp_failure) > 0:
                ptp_rate = ptp_success / (ptp_success + ptp_failure)

        # 汇总 rate 列表
        if ftp_rate is not None:
            ftp_rates.append(ftp_rate)
        if ptp_rate is not None:
            ptp_rates.append(ptp_rate)

        # 计算 resolved
        if inst_report.get(instance_id, {}).get("resolved", False):
            resolved_count += 1

    # 汇总输出
    print('total_count:', total_count)
    metrics = {
        '%Resolved': (resolved_count / total_count * 100.0) if total_count else None,
        '%Apply': (apply_count / total_count * 100.0) if total_count else None,
        'avg_FTP_rate': sum(ftp_rates) / len(ftp_rates) if ftp_rates else None,
        'avg_PTP_rate': sum(ptp_rates) / len(ptp_rates) if ptp_rates else None
    }
    return metrics



@ICL_EVALUATORS.register_module()
class SWEBenchEvaluator(BaseEvaluator):
    """Evaluator for SWE-bench."""

    def __init__(self,
                 timeout: int = 1800,
                 log_dir: str = '/outputs/swebench_eval',
                 build_docker_images: bool = False,
                 pull_remote_images_if_available: bool = True,
                 force_arch: str = '',
                 **kwargs):
        """
        Args:
            timeout: 单个任务的评测超时秒数
            log_dir: 日志输出目录
            build_docker_images: 是否本地构建docker镜像（否则直接拉取）
            pull_remote_images_if_available: 启动前尝试从dockerhub拉取
            force_arch: 可选强制架构('arm64'或'x86_64')
        """
        super().__init__()
        self.timeout = timeout
        self.log_dir = log_dir
        self.build_docker_images = build_docker_images
        self.pull_remote_images_if_available = pull_remote_images_if_available
        self.force_arch = force_arch

        
    def score(self, predictions: List[Any], references: List[Any], test_set: Any):
        """OpenCompass会把模型预测、参考标签和HF Dataset传进来."""
        if len(predictions) != len(references):
            return {'error': 'predictions and references length mismatch'}

        # 如果需要，可以调用 build_images() 构建镜像
        if self.build_docker_images:
            from opencompass.datasets.SWE_bench.build_images import build_images
            samples = test_set['test'] if 'test' in test_set else test_set
            # check_data(samples)
            # 构建镜像
            build_images(samples=samples, 
                         force_rebuild=False,
                         max_workers=4,
                         use_remote_images=self.pull_remote_images_if_available,
                         force_arch=self.force_arch)
        
        # 遍历模型输出，逐条评测
        details = {}
        resolved_count = 0
        total_count = len(predictions)
        report_all = {}
        for idx, (pred, instance_id) in enumerate(zip(predictions, references)):
            # 1. 清理模型输出成git diff patch
            patch = extract_diff(pred)
            # patch = find_golden_patch(instance_id=instance_id) # 替换为golden patch进行测试

            # 2. 从数据集中找到这个instance的metadata
            if 'test' in test_set:
                matched = [m for m in test_set['test'] if m['instance_id'] == instance_id]
            else:
                matched = [m for m in test_set if m['instance_id'] == instance_id]

            if not matched:
                details[str(idx)] = {'error': 'Instance metadata not found', 'resolved': 0}
                continue

            instance_meta = matched[0]

            # 3. 调用eval_instance进行真实环境评测
            eval_result = eval_instance(instance=instance_meta,
                                        pred=patch,
                                        timeout=self.timeout,
                                        log_dir=self.log_dir)
            
            
            resolved_flag = 1 if eval_result.get('resolved', False) else 0
            resolved_count += resolved_flag
            report_all[instance_id] = eval_result.get('report', {})

            details[str(idx)] = {
                'resolved': resolved_flag,
                'completed': eval_result.get('completed', False),
                'report': eval_result.get('report', {})
            }


        metrics_dict = compute_swebench_metrics(report_all)
        metrics_dict['details'] = details
        return metrics_dict
