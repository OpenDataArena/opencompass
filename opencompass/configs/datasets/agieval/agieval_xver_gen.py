from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import AGIEvalDataset_v2, AGIEvalEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess, first_capital_postprocess_multi
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocessx as generic_llmjudge_postprocess
with read_base():
    from opencompass.configs.models.qwen2_5.lmdeploy_xVerify_9b import (
        models as judge_model,
    )
agieval_reader_cfg = dict(
    input_columns=['question', 'options'], output_column='label')

agieval_single_choice_sets = [
    'gaokao-chinese',
    'gaokao-english',
    'gaokao-geography',
    'gaokao-history',
    'gaokao-biology',
    'gaokao-chemistry',
    'gaokao-mathqa',
    'logiqa-zh',
    'lsat-ar',
    'lsat-lr',
    'lsat-rc',
    'logiqa-en',
    'sat-math',
    'sat-en',
    'sat-en-without-passage',
    'aqua-rat',
]
agieval_multiple_choices_sets = [
    'gaokao-physics',
    'jec-qa-kd',
    'jec-qa-ca',
]
agieval_cloze_sets = ['gaokao-mathcloze', 'math']
agieval_chinese_sets = [
    'gaokao-chinese',
    'gaokao-english',
    'gaokao-geography',
    'gaokao-history',
    'gaokao-biology',
    'gaokao-chemistry',
    'gaokao-physics',
    'gaokao-mathqa',
    'logiqa-zh',
    'gaokao-mathcloze',
    'jec-qa-kd',
    'jec-qa-ca',
]
agieval_english_sets = [
    'lsat-ar',
    'lsat-lr',
    'lsat-rc',
    'logiqa-en',
    'sat-math',
    'sat-en',
    'sat-en-without-passage',
    'aqua-rat',
    'math',
]
agieval_gaokao_sets = [
    'gaokao-chinese',
    'gaokao-english',
    'gaokao-geography',
    'gaokao-history',
    'gaokao-biology',
    'gaokao-chemistry',
    'gaokao-physics',
    'gaokao-mathqa',
]
GRADER_TEMPLATE = """ 
    -
    Special considerations:

    1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

    2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

    3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content, respond with [Correct].

    4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
    -

    Question: {{question}}\n{{options}}\n

    Output sentence: {prediction}

    Correct answer: {label}

    Judgement:
""".strip()
agieval_datasets = []
for _name in agieval_single_choice_sets:
    if _name in agieval_chinese_sets:
        _hint = '答案是： '
    else:
        _hint = 'The answer is '
    agieval_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN', prompt=f'{{question}}\n{{options}}\n{_hint}')
            ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=4096))

    agieval_eval_cfg = dict(
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
                type=AGIEvalDataset_v2,
                path='opencompass/agieval',
                name=_name,
                abbr='agieval-' + _name,
                setting_name='zero-shot',
                reader_cfg=agieval_reader_cfg,
            ),
            judge_cfg=judge_model[0],
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
        pred_role='BOT',
    )

    agieval_datasets.append(
        dict(
            type=AGIEvalDataset_v2,
            path='opencompass/agieval',
            name=_name,
            abbr='agieval-' + _name,
            setting_name='zero-shot',
            reader_cfg=agieval_reader_cfg,
            infer_cfg=agieval_infer_cfg.copy(),
            eval_cfg=agieval_eval_cfg.copy()))

for _name in agieval_multiple_choices_sets:
    if _name in agieval_chinese_sets:
        _hint = '答案是： '
    else:
        _hint = 'The answer is '
    agieval_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN', prompt=f'{{question}}\n{{options}}\n{_hint}')
            ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=4096))

    agieval_eval_cfg = agieval_eval_cfg = dict(
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
                type=AGIEvalDataset_v2,
                path='opencompass/agieval',
                name=_name,
                abbr='agieval-' + _name,
                setting_name='zero-shot',
                reader_cfg=agieval_reader_cfg,
            ),
            judge_cfg=judge_model[0],
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
        pred_role='BOT',
    )

    agieval_datasets.append(
        dict(
            type=AGIEvalDataset_v2,
            path='opencompass/agieval',
            name=_name,
            abbr='agieval-' + _name,
            setting_name='zero-shot',
            reader_cfg=agieval_reader_cfg,
            infer_cfg=agieval_infer_cfg.copy(),
            eval_cfg=agieval_eval_cfg.copy()))

for _name in agieval_cloze_sets:
    if _name in agieval_chinese_sets:
        _hint = '答案是： '
    else:
        _hint = 'The answer is '
    agieval_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt=f'{{question}}\n{_hint}')])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=4096))

    agieval_eval_cfg = agieval_eval_cfg = dict(
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
                type=AGIEvalDataset_v2,
                path='opencompass/agieval',
                name=_name,
                abbr='agieval-' + _name,
                setting_name='zero-shot',
                reader_cfg=agieval_reader_cfg,
            ),
            judge_cfg=judge_model[0],
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
        pred_role='BOT',
    )

    agieval_datasets.append(
        dict(
            type=AGIEvalDataset_v2,
            path='opencompass/agieval',
            name=_name,
            abbr='agieval-' + _name,
            setting_name='zero-shot',
            reader_cfg=agieval_reader_cfg,
            infer_cfg=agieval_infer_cfg.copy(),
            eval_cfg=agieval_eval_cfg.copy()))

for _item in agieval_datasets:
    _name = _item['name']
    _intro = {
        'gaokao-chinese':
        '以下是一道中国高考语文选择题，请选择正确的答案。',
        'gaokao-english':
        '以下是一道中国高考英语选择题，请选择正确的答案。',
        'gaokao-geography':
        '以下是一道中国高考地理选择题，请选择正确的答案。',
        'gaokao-history':
        '以下是一道中国高考历史选择题，请选择正确的答案。',
        'gaokao-biology':
        '以下是一道中国高考生物选择题，请选择正确的答案。',
        'gaokao-chemistry':
        '以下是一道中国高考化学选择题，请选择正确的答案。',
        'gaokao-physics':
        '以下是一道中国高考物理选择题，请选择正确的答案。',
        'gaokao-mathqa':
        '以下是一道中国高考数学选择题，请选择正确的答案。',
        'logiqa-zh':
        '以下是一道中国公务员考试题，请选择正确的答案。',
        'lsat-ar':
        'The following is a LSAT Analytical Reasoning question. Please select the correct answer.',
        'lsat-lr':
        'The following is a LSAT Logical Reasoning question. Please select the correct answer.',
        'lsat-rc':
        'The following is a LSAT Reading Comprehension question. Please select the correct answer.',
        'logiqa-en':
        'The following is a Logic Reasoning question. Please select the correct answer.',
        'sat-math':
        'The following is a SAT Math question. Please select the correct answer.',
        'sat-en':
        'The following is a SAT English question. Please select the correct answer.',
        'sat-en-without-passage':
        'The following is a SAT English question. Please select the correct answer.',
        'aqua-rat':
        'The following is a AQUA-RAT question. Please select the correct answer.',
        'jec-qa-kd':
        '以下是一道中国司法考试基础知识题，请选择正确的答案。',
        'jec-qa-ca':
        '以下是一道中国司法考试案例分析题，请选择正确的答案。',
        'gaokao-mathcloze':
        '以下是一道中国高考数学填空题，请填入正确的答案。',
        'math':
        'The following is a Math question. Please select the correct answer.',
    }[_name]
    _templates = _item['infer_cfg']['prompt_template']['template']
    _templates['round'][0][
        'prompt'] = _intro + '\n' + _templates['round'][0]['prompt']

del _item, _intro, _templates, _name, _hint, agieval_infer_cfg, agieval_eval_cfg
