"""
export PYTHONPATH=/home/featurize/opencompass 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python run.py examples/eval_qwen2_5_coder_1_5_b_swebench.py -a vllm  --debug
"""

from mmengine.config import read_base

with read_base():
    from ..opencompass.configs.datasets.swe_bench.swe_bench_bm25_27k import \
        swebench_datasets
    from ..opencompass.configs.models.qwen2_5.vllm_qwen2_5_coder_1_5b_instruct import \
        models as vllm_qwen2_5_coder_1_5b_instruct_models

datasets = swebench_datasets
models = vllm_qwen2_5_coder_1_5b_instruct_models
