from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm2-chat-7b-hf',
        path='/mnt/dhwfile/raise/user/caimengzhang/model/models--internlm--internlm2-chat-7b/snapshots/e7c2e16310627a098500e3ca30eaf4cd2690b9fc',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
