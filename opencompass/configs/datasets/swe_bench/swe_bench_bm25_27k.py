from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.swe_bench import SWEBenchDataset  # 或者 HuggingFace原生Dataset

swebench_reader_cfg = dict(
    input_columns=['text'], # 使用BM25版的 text 字段
    output_column='instance_id', # golden patch, 但在评测中不需要
    train_split='test',
    test_split='test',
    test_range=1
)


swebench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{text}'),   # 模型直接看到完整prompt
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048)  # patch可能很长
)


swebench_eval_cfg = dict(
    evaluator=dict(type='SWEbenchEvaluator', 
                   timeout=1800,
                   log_dir = '/home/featurize/opencompass/outputs/swebench_eval',
                   build_docker_images = True,
                   pull_remote_images_if_available = True,
                   ),  # 这里挂你写的评估器类
    pred_role='BOT'
)


swebench_datasets = [
    dict(
        type=SWEBenchDataset,     # 或者自定义Dataset类
        abbr='swebench-bm25-27k',
        path='/home/featurize/data/princeton-nlp/SWE-bench_bm25_27K',
        reader_cfg=swebench_reader_cfg,
        infer_cfg=swebench_infer_cfg,
        eval_cfg=swebench_eval_cfg
    )
]
