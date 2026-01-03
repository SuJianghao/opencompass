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


class Gsm8kEvaluator(BaseEvaluator):

    def is_equal(self, pred, refer):
        try:
            if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
                return True
        except Exception:
            pass
        return False

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if self.is_equal(i, j):
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result


class Gsm8kAgentEvaluator(BaseEvaluator):
    """Gsm8k agent evaluator for soft condition.

    Args:
        action (str): Action for catching internal prediction.
            Defaults to `PythonInterpreter`.
    """

    def __init__(self, action: str = 'PythonInterpreter'):
        self.action = action

    def is_equal(self, pred, refer):
        try:
            if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
                return True
        except Exception:
            pass
        return False

    def soft_equal(self, pred, refer, step):
        try:
            soft_pred = step['result']['text']
            if abs(float(soft_pred) - int(refer)) < 1e-6:
                return True
        except Exception:
            # result might not exists
            # text cannot convert to float
            pass
        return False

    def get_action(self, step):
        for s in step[::-1]:
            if s['type'] == self.action:
                return s

    def score(self, predictions, references, steps):
        """Calculate accuracy."""
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        row_reasoning_scope = 0
        action_scope = 0
        code_scope = 0
        reasoning_scope = 0
        final_scope = 0
        total = len(references)
        for pred, refer, step in zip(predictions, references, steps):
            # if final answer right
            if self.is_equal(pred, refer):
                if self.get_action(step):
                    final_scope += 1
                else:
                    row_reasoning_scope += 1
            else:
                s = self.get_action(step)
                if s:
                    action_scope += 1
                    if not s['errmsg']:
                        code_scope += 1
                        # whether action result is correct
                        reasoning_scope += self.soft_equal(pred, refer, s)

        result = dict(
            follow_acc=100 * (row_reasoning_scope + final_scope) / total,
            reasoning_acc=100 *
            (reasoning_scope + final_scope + row_reasoning_scope) / total,
            code_acc=100 * (code_scope + final_scope) / total,
            action_pct=100 * (action_scope + final_scope) / total,
        )
        return result
