import re

from opencompass.utils import get_logger


def get_final_results(judged_answers,
                      references,
                      origial_responses,
                      metric_name='accuracy'):
    count = 0
    is_correct_count = 0
    is_incorrect_count = 0
    is_not_attempted_count = 0
    attempted_judge_count = 0
    details = []
    for i, j, k in zip(judged_answers, references, origial_responses):
        if i in ['A', 'B']:
            attempted_judge_count += 1
        grade_letter = i
        detail = {
            'pred': k,
            'ref': j,
            'origin_grade_response': i,
            'grade_letter': grade_letter,
            'correct': False,
        }
        count += 1
        if grade_letter == 'A':
            is_correct_count += 1
            detail['correct'] = True
        elif grade_letter == 'B':
            is_incorrect_count += 1
        else:
            is_not_attempted_count += 1
        details.append(detail)

    is_correct = is_correct_count / count
    is_incorrect = is_incorrect_count / count
    is_given_attempted = is_correct + is_incorrect
    accuracy_given_attempted = (is_correct / is_given_attempted
                                if is_given_attempted > 0 else 0)
    attempted_judge_ratio = attempted_judge_count / count

    f1 = (2 * accuracy_given_attempted * is_correct /
          (accuracy_given_attempted + is_correct) if
          (accuracy_given_attempted + is_correct) > 0 else 0)
    result = {
        metric_name: is_correct * 100,
        f'{metric_name}_given_attempted': accuracy_given_attempted * 100,
        'f1': f1,
        'attempted_ratio': attempted_judge_ratio * 100,
        'correct_count': is_correct_count,
        'incorrect_count': is_incorrect_count,
        'not_attempted_count': is_not_attempted_count,
        'details': details,
    }
    return result


def _generic_llmjudge_postprocess(judgement: str):
    match = re.search(r'(A|B)', judgement)
    grade_letter = (match.group(0) if match else 'unknown'
                    )  # Return 'unknown' if no match
    return grade_letter


def generic_llmjudge_postprocess(
    output: dict,
    output_path: str,
) -> dict:
    judged_answers = []
    origial_responses = []
    references = []
    for k, v in output.items():
        origial_responses.append(v['prediction'])
        processed_judge = _generic_llmjudge_postprocess(v['prediction'])
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            try:
                references.append(v['gold'])

            except KeyError:
                get_logger().warning(
                    f'No gold answer for {k}, use empty string as reference!')
                references.append('')
    results = get_final_results(judged_answers, references, origial_responses)
    results['details'] = output
    return results

def _generic_llmjudge_postprocessx(judgement: str):
    #match = re.search(r'(A|B)', judgement)
    #match = re.search(r'(Correct|Incorrect)', judgement)
    if judgement: # 确保字符串不是 None 或空
        clean_judgement = judgement.strip()
        match = re.search(r'\b(Correct|Incorrect)\b', clean_judgement, re.IGNORECASE)
        if match:
            result = match.group(1).capitalize() # 统一格式为 'Correct' 或 'Incorrect'
            #print(f"最终判断结果: {result}")
        else:
            print("在清理后的字符串中未找到有效判断。")
    else:
        print("输入的 judgement 为空。")
        return 'unknown'
    result = 'unknown' 
    if match:
    # 2. 获取匹配到的单词 (e.g., 'Correct', 'incorrect', 'CORRECT', etc.)
        matched_word = match.group(1)
        
        # 3. 判断匹配到的单词，并转换为 'A' 或 'B'
        #    使用 .lower() 来统一处理大小写，确保逻辑正确
        if matched_word.lower() == 'correct':
            result = 'A'
        elif matched_word.lower() == 'incorrect':
            result = 'B'
    else:
        # 如果没有匹配到 "Correct" 或 "Incorrect"
        result = 'unknown' # 或者可以设置为 'C'、'D' 等表示其他情况
    return result

def generic_llmjudge_postprocessx(
    output: dict,
    output_path: str,
) -> dict:
    judged_answers = []
    origial_responses = []
    references = []
    for k, v in output.items():
        origial_responses.append(v['prediction'])
        processed_judge = _generic_llmjudge_postprocessx(v['prediction'])
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            try:
                references.append(v['gold'])

            except KeyError:
                get_logger().warning(
                    f'No gold answer for {k}, use empty string as reference!')
                references.append('')
    results = get_final_results(judged_answers, references, origial_responses)
    results['details'] = output
    return results

def generic_llmjudge_academic_postprocess(
    output: dict,
    output_path: str,
    metric_name: str = 'accuracy',
) -> dict:
    judged_answers = []
    origial_responses = []
    references = []
    for k, v in output.items():
        origial_responses.append(v['prediction'])
        processed_judge = _generic_llmjudge_postprocess(v['prediction'])
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            references.append(v['gold'])
    results = get_final_results(judged_answers, references, origial_responses,
                                metric_name)
    results['details'] = output
    # For academic summarizer
    results.pop('f1', None)
    return results
