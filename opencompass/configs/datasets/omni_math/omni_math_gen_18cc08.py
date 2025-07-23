from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

from opencompass.datasets.omni_math import OmniMathDataset, OmniMathEvaluator


reader_cfg = dict(
    input_columns=['problem'], 
    output_column='answer'
)

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='please answer the following mathematical question, put your final answer in \\boxed{}.\n\n{problem}'),
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        temperature=0.0
    )
)

eval_cfg = dict(
    evaluator=dict(
        type=OmniMathEvaluator,
        url=['http://10.140.24.11:8002','http://10.140.24.11:8003']
    )
)

omni_math_datasets = [
    dict(
        type=OmniMathDataset,
        abbr='OmniMath',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg
    )
]