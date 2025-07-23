# USED IN BASE MODEL
from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import DropOpenAIDataset, DropOpenAIEvaluaor
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocessx as generic_llmjudge_postprocess
with read_base():
    from opencompass.configs.models.qwen2_5.lmdeploy_xVerify_9b import (
        models as judge_model,
    )
    from .drop_examples import drop_examples  # noqa: F401, F403

drop_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='answers',
    train_split='validation',
    test_split='validation',
)

template = f'''\
You will be asked to read a passage and answer a question. Think step by step, then write a line of the form "Answer: $ANSWER" at the end of your response. Some examples of passages and Q&A are provided below.

{drop_examples}

# Your Task

---
{{prompt}}'''

drop_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template=dict(round=[dict(role='HUMAN', prompt=template)])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, stopping_criteria=['---', 'Passage', 'Question', 'You will be asked']),)
GRADER_TEMPLATE = """ 
    -
    Special considerations:

    1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

    2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

    3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content, respond with [Correct].

    4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
    
    5. **Multiple Correct Answers**: If there are multiple correct answers, as long as the Output sentence contains one of the correct answers, respond with [Correct]. 
    -

    Question: {prompt}

    Output sentence: {prediction}

    Correct answer: {answers}

    Judgement:
""".strip()
drop_eval_cfg = dict(
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
                type=DropOpenAIDataset,
                path='data/drop_simple_eval/dev.jsonl',
                reader_cfg=drop_reader_cfg,
            ),
            judge_cfg=judge_model[0],
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
        pred_role='BOT',
    )

drop_datasets = [
    dict(
        abbr='drop',
        type=DropOpenAIDataset,
        path='data/drop_simple_eval/dev.jsonl',
        reader_cfg=drop_reader_cfg,
        infer_cfg=drop_infer_cfg,
        eval_cfg=drop_eval_cfg)
]
