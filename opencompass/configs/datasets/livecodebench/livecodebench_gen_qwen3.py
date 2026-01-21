from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    LCBCodeGenerationDataset,
    LCBCodeGenerationEvaluator,
)
from opencompass.datasets.livecodebench import TestOutputPromptConstants


lcb_code_generation_reader_cfg = dict(
    input_columns=[
        'question_content',
        'format_prompt',
    ],
    # output_column='evaluation_sample',
    output_column='question_id',
)

SYSTEM_MESSAGE_GENERIC = f'<|im_start|>system\nYou are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.<|im_end|>\n<|im_start|>user'

prompt_template = '### Question:\n{question_content}\n\n{format_prompt}'


# Code Generation Tasks
lcb_code_generation_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
             begin=[
                 dict(
                     role='SYSTEM',
                     prompt=SYSTEM_MESSAGE_GENERIC
                 ),
            ],
            round=[
                dict(
                    role='HUMAN',
                    prompt=prompt_template
                )
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

lcb_code_generation_eval_cfg = dict(
    evaluator=dict(
        type=LCBCodeGenerationEvaluator,
        num_process_evaluate=4,
        timeout=6,
        release_version='release_v5',
                   start_date='2024-08-01',
                   end_date='2025-02-01'
    ),
    pred_role='BOT',
)

LCBCodeGeneration_dataset = dict(
    type=LCBCodeGenerationDataset,
    abbr='lcb_code_generation',
    path='opencompass/code_generation_lite',
    reader_cfg=lcb_code_generation_reader_cfg,
    infer_cfg=lcb_code_generation_infer_cfg,
    eval_cfg=lcb_code_generation_eval_cfg,
    release_version='release_v5',
                   start_date='2024-08-01',
                   end_date='2025-02-01'
)





LCB_datasets = [
    LCBCodeGeneration_dataset,
]
