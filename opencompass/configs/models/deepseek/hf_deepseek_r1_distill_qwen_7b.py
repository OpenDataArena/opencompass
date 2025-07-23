from opencompass.models import HuggingFacewithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content
from opencompass.models import (
    TurboMindModelwithChatTemplate,
)
models = [
    #dict(
    #    type=HuggingFacewithChatTemplate,
    #    abbr='deepseek-r1-distill-qwen-7b-hf',
    #    path='/mnt/dhwfile/raise/user/caimengzhang/model/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60',
    #    max_out_len=16384,
    #    batch_size=8,
    #    run_cfg=dict(num_gpus=1),
    #    pred_postprocessor=dict(type=extract_non_reasoning_content)
    #),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-r1-distill-qwen-7b-turbomind',
        path='/mnt/dhwfile/raise/user/caimengzhang/model/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60',
        engine_config=dict(session_len=131072, max_batch_size=128, tp=1),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=131072,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=1, output_subdir="raw_predictions"),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
