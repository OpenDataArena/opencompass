# THIS SHALL ALSO BE DEPRECATED
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalDataset, HumanEvalPlusEvaluator, humaneval_postprocess_v2

humaneval_plus_reader_cfg = dict(
    input_columns=['prompt'], output_column='task_id', train_split='test')

# TODO: allow empty output-column
humaneval_plus_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                #prompt='Please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:\n{prompt}\nBelow is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests: '),
                #prompt='Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:\n{prompt}\nBelow is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:'),
                #prompt='Please provide a self-contained Python script that solves the following problem in a markdown code block:\n{prompt}\nBelow is a Python script with a self-contained function that solves the problem and passes corresponding tests:'),
                prompt='Please provide an efficient and self-contained Python script that solves the following problem in a markdown code block, Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:\n{prompt}'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

humaneval_plus_eval_cfg = dict(
    evaluator=dict(type=HumanEvalPlusEvaluator),
    pred_role='BOT',
    k=[1, 10, 100],  # the parameter only for humaneval
    pred_postprocessor=dict(type=humaneval_postprocess_v2),
)

humaneval_plus_datasets = [
    dict(
        abbr='humaneval_plus',
        type=HumanevalDataset,
        path='opencompass/humaneval',
        reader_cfg=humaneval_plus_reader_cfg,
        infer_cfg=humaneval_plus_infer_cfg,
        eval_cfg=humaneval_plus_eval_cfg)
]
