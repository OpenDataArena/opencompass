source activate opencompass
sleep 2


if [ $# -eq 0 ]; then
  echo "请在命令行中提供模型地址，例如：llama3.1-8b/full/sft-gsm8k 以及数据集名称 如：gsm8k"
  exit 1
fi
START_TIME=`date +%Y%m%d-%H:%M:%S`
MODEL_PATH=$1
DATANAME=$2
EXP_NAME=llama_test
output_file="opencompass/configs/models/hf_llama/hf_llama3_1_8b_instruct_${START_TIME}.py"
LOG_FILE=logs/${START_TIME}_${EXP_NAME}.log
cat << EOF > $output_file
from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='llama-3_1-8b-instruct-vllm',
        path='${MODEL_PATH}',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=32768,
        batch_size=8,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
EOF
echo "YAML 配置文件已成功创建：$output_file"


python replace_chat_template_to_default.py --model_path ${MODEL_PATH} 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 opencompass \
    --max-out-len 32768 \
    -w outputs/${DATANAME}_${START_TIME}_${EXP_NAME} \
    --datasets \
    omni_math_gen OlympiadBenchMath_0shot_xver_gen \
    gsm8k_0shot_xver_gen math_0shot_xver_gen math_prm800k_500_0shot_cot_xver_gen aime2024_repeat8_xver_gen \
    --hf-type chat --models hf_llama3_1_8b_instruct_${START_TIME} --max-num-worker 8 -a vllm
