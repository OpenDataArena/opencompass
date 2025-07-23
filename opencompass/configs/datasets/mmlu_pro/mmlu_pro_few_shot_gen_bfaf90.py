from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MMLUProDataset, MMLUProBaseEvaluator
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import match_answer_pattern
with read_base():
    from .mmlu_pro_categories import categories

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
            inferencer=dict(type=GenInferencer,max_out_len=2048,stopping_criteria=['Question:'])
    )

    # mmlu_pro_eval_cfg = dict(
    #     evaluator=dict(type=MMLUProBaseEvaluator)
    # )
    mmlu_pro_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(
            type=match_answer_pattern,
            answer_pattern=r'(?i)The answer is\s*([A-P])')
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
