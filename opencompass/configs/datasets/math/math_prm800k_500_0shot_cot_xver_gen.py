from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import MATHEvaluator
from opencompass.datasets import (
    MATHDataset,
    math_postprocess_v2,
    normalize_final_answer,
)
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocessx as generic_llmjudge_postprocess
with read_base():
    from opencompass.configs.models.qwen2_5.lmdeploy_xVerify_9b import (
        models as judge_model,
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
    inferencer=dict(type=GenInferencer),
)

GRADER_TEMPLATE = """ 
    -
    Special considerations:

    1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

    2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

    3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content, respond with [Correct].

    4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
    -

    Question: {question}

    Output sentence: {prediction}

    Correct answer: {solution}

    Judgement:
""".strip()
math_eval_cfg = dict(
        evaluator=dict(
            type=GenericLLMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt="You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect]."
                        #prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs."
                        )
                ],
                    round=[
                    dict(
                        role='HUMAN',
                        prompt = GRADER_TEMPLATE
                    ),
                ]),
            ),
            dataset_cfg=dict(
                type=MATHDataset,
                path='opencompass/math',
                file_name='test_prm800k_500.json',
                reader_cfg=math_reader_cfg,
            ),
            judge_cfg=judge_model[0],
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
        pred_role='BOT',
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
