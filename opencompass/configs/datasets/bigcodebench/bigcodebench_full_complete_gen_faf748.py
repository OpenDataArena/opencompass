from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (BigCodeBenchDataset, BigCodeBenchEvaluator)

bigcodebench_full_reader_cfg = dict(
    input_columns=['complete_prompt'],
    output_column='test',
)

bigcodebench_full_infer_cfg = dict(prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        begin=[dict(role='system', fallback_role='HUMAN', prompt='')],
        round=[
            dict(role='HUMAN', prompt='{complete_prompt}'),
        ])),
                                   retriever=dict(type=ZeroRetriever),
                                   inferencer=dict(type=GenInferencer,
                                                   max_out_len=1024))

bigcodebench_full_eval_cfg = dict(
    evaluator=dict(
        type=BigCodeBenchEvaluator,
        release_version='v0.1.2',
        eval_type='complete',
        # remote_execute_api='https://bigcode-bigcodebench-evaluator.hf.space/',
        remote_execute_api=
        'https://sd07o2i9k9hp5cpdkk6og.apigateway-cn-beijing.volceapi.com/mlp/s-20250428212320-gzwcq/',  # noqa: E501
        dataset_version='full',
    ),
    pred_role='BOT',
)

bigcodebench_full_complete_datasets = [
    dict(abbr='bigcodebench_full_complete',
         type=BigCodeBenchDataset,
         path='opencompass/bigcodebench',
         reader_cfg=bigcodebench_full_reader_cfg,
         infer_cfg=bigcodebench_full_infer_cfg,
         eval_cfg=bigcodebench_full_eval_cfg,
         release_version='v0.1.2')
]
