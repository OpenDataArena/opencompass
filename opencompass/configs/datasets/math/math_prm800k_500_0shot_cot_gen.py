from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    MATHDataset,
    MATHEvaluator,
    math_postprocess_v2,
    normalize_final_answer,
)

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')
math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.',
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer,max_out_len=4096),
)

# postprocess v2
math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'),
    pred_postprocessor=dict(type=math_postprocess_v2),
)

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math_prm800k_500',
        path='opencompass/math',
        file_name='test_prm800k_500.json',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]