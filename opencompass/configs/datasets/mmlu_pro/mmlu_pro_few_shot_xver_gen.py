from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MMLUProDataset, MMLUProBaseEvaluator
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import match_answer_pattern
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocessx as generic_llmjudge_postprocess
with read_base():
    from .mmlu_pro_categories import categories
    from opencompass.configs.models.qwen2_5.lmdeploy_xVerify_9b import (
        models as judge_model,
    )
    

mmlu_pro_datasets = []

for category in categories:
    hint = f'The following are multiple choice questions (with answers) about {category}, Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.'
    question_and_options = 'Question:\n{question}\nOptions:\n{options_str}'
    cot_content='{cot_content}' 
    mmlu_pro_reader_cfg = dict(
        input_columns=['question', 'cot_content', 'options_str'],
        output_column='answer',
        train_split='validation',
        test_split='test',
    )
    mmlu_pro_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=f'{question_and_options}\nAnswer: Let\'s think step by step. {cot_content} The answer is {{answer}}'),
        prompt_template=dict(
            type=PromptTemplate,
            template=f'{hint}\n</E>{question_and_options}\nAnswer: Let\'s think step by step. ',
            ice_token='</E>'
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
            #inferencer=dict(type=GenInferencer, max_out_len=2048,stop_words=['Question:']) 
            inferencer=dict(type=GenInferencer,stopping_criteria=['Question:'])
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

    Correct answer: {answer}

    Judgement:
""".strip()
    # mmlu_pro_eval_cfg = dict(
    #     evaluator=dict(type=MMLUProBaseEvaluator)
    # )
    # mmlu_pro_eval_cfg = dict(
    #     evaluator=dict(type=AccEvaluator),
    #     pred_postprocessor=dict(
    #         type=match_answer_pattern,
    #         answer_pattern=r'(?i)(?:The\s+(?:correct\s+)?answer\s+is|ANSWER)\s*[:：]?\s*(?:[（(])?\s*([A-P])')
    # )
    
    mmlu_pro_eval_cfg = dict(
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
                type=MMLUProDataset,
                category=category,
                path='opencompass/mmlu_pro',
                reader_cfg=mmlu_pro_reader_cfg,
            ),
            judge_cfg=judge_model[0],
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
        pred_role='BOT',
    )


    mmlu_pro_datasets.append(
        dict(
            abbr=f'mmlu_pro_{category.replace(" ", "_")}',
            type=MMLUProDataset,
            path='opencompass/mmlu_pro',
            category=category,
            reader_cfg=mmlu_pro_reader_cfg,
            infer_cfg=mmlu_pro_infer_cfg,
            eval_cfg=mmlu_pro_eval_cfg,
        ))
