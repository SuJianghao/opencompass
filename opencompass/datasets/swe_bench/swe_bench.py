import json
import os
import re
from os import environ

import datasets as hf_datasets
from datasets import Dataset, DatasetDict
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from ..base import BaseDataset

from pathlib import Path
from typing import List, Any

from opencompass.openicl.icl_evaluator.base import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS

from opencompass.datasets.swe_bench.utils import eval_instance
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

        train_files = collect_files('train')
        dev_files = collect_files('dev')
        test_files = collect_files('test')

        dataset = hf_datasets.load_dataset(
            'json',  # 同时支持 .json 和 .jsonl
            data_files={
                'train': train_files,
                'dev': dev_files,
                'test': test_files
            }
        )
        return dataset



@TEXT_POSTPROCESSORS.register_module('gsm8k_dataset')
def gsm8k_dataset_postprocess(text: str) -> str:
    return text.split('#### ')[1].replace(',', '')



@TEXT_POSTPROCESSORS.register_module('gsm8k')
def gsm8k_postprocess(text: str) -> str:
    text = text.split('Question:')[0]
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    if not numbers:
        return 'NULL'
    return numbers[-1]



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
            from opencompass.datasets.swe_bench.build_images import build_images
            samples = test_set['test'] if 'test' in test_set else test_set
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

        for idx, (pred, instance_id) in enumerate(zip(predictions, references)):
            # 1. 清理模型输出成git diff patch
            patch = extract_diff(pred)

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

            details[str(idx)] = {
                'resolved': resolved_flag,
                'completed': eval_result.get('completed', False),
                'report': eval_result.get('report', {})
            }

        acc = (resolved_count / total_count * 100.0) if total_count > 0 else 0.0
        return {
            'acc': acc,
            'details': details
        }
