from mmengine.config import read_base

with read_base():
    from .swe_bench_bm25_13k import swebench_datasets # noqa: F401, F403
