from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MMLUProDataset, MMLUProBaseEvaluator
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import match_answer_pattern
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess
with read_base():
    from .mmlu_pro_categories import categories
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_32b_instruct import (
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
#     GRADER_TEMPLATE = """
#     Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
    
#     Here are some evaluation criteria:
#     1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
#     2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
#     3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
#     4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
#     5. If the prediction is given with \\boxed{}, please ignore the \\boxed{} and only judge whether the candidate's answer is consistent with the standard answer.

#     Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
#     A: CORRECT 
#     B: INCORRECT
#     Just return the letters "A" or "B", with no text around it.

#     Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.


#     <Original Question Begin>: \n{question}\n<Original Question End>\n\n
#     <Gold Target Begin>: \n{answer}\n<Gold Target End>\n\n
#     <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n
    
#     Judging the correctness of candidates' answers:
# """.strip()
    GRADER_TEMPLATE = """
    Evaluation Criteria

    Please select your judgment from the two options below.

    A. Correct
    Choose A if any of the following conditions are met:
    Primary Rule:
    If the question provides explicit options (e.g., A, B, C, D) and the model selects the correct option, you must choose A, regardless of the quality of its reasoning.
    For questions without explicit options, the model’s final answer aligns with the correct answer in content and meaning.
    The model’s answer is mathematically or logically equivalent to the correct answer, despite differences in formatting (e.g., 50% vs. 0.5, 5 PM vs. 17:00).
    If the model provides multiple answers or self-corrects, its final and conclusive answer is used for the evaluation, and that final answer is correct.
    
    B. Incorrect
    Choose B if any of the following conditions are met:

    The model’s final answer is factually incorrect.
    In a multiple-choice question, the model selects the wrong option.
    The model’s answer is ambiguous, incomplete, or fails to directly answer the question.
    Even if parts of the reasoning are correct, the final stated answer or conclusion is incorrect.

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
                        prompt="You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either A or B. A: CORRECT  B: INCORRECT"
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
