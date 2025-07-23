from opencompass.datasets.korbench.korbench import korbenchDataset, korbenchEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocessx as generic_llmjudge_postprocess
with read_base():
    from opencompass.configs.models.qwen2_5.lmdeploy_xVerify_9b import (
        models as judge_model,
    )
categories = ['cipher', 'counterfactual', 'logic', 'operation', 'puzzle']

korbench_0shot_single_datasets = []

for category in categories:
    # Prompt template
    prompt_template = dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='HUMAN',
                    prompt=''
                )
            ],
            round=[
                dict(
                    role='HUMAN',
                    prompt='{prompt}' # f-string
                )
            ]
        )
    )

    # Reader configuration
    reader_cfg = dict(
        input_columns=['prompt'],
        output_column='answer',
    )

    # Inference configuration
    infer_cfg = dict(
        prompt_template=prompt_template,
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

    Question: {prompt}

    Output sentence: {prediction}

    Correct answer: {answer}

    Judgement:
""".strip()
    # Evaluation configuration
    eval_cfg =dict(
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
                type=korbenchDataset,
                abbr=f'korbench_{category}',
                path='opencompass/korbench',
                prompt_mode='0_shot',
                category=category,
                reader_cfg=reader_cfg,
            ),
            judge_cfg=judge_model[0],
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
        pred_role='BOT',
    ) 

    korbench_dataset = dict(
        type=korbenchDataset,
        abbr=f'korbench_{category}',
        path='opencompass/korbench',
        prompt_mode='0_shot',
        category=category,
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )

    korbench_0shot_single_datasets.append(korbench_dataset)
