import json
import os
import argparse

# add model pah as argument
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
args = parser.parse_args()

# load tokenizer config
tokenizer_config_path = os.path.join(args.model_path, 'tokenizer_config.json')
tokenizer_config = json.load(open(tokenizer_config_path, encoding='utf-8'))

llamafact_default_chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}{% if system_message is defined %}{{ system_message + '\n' }}{% endif %}{% for message in loop_messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ 'Human: ' + content + '\nAssistant:' }}{% elif message['role'] == 'assistant' %}{{ content + '<|endoftext|>' + '\n' }}{% endif %}{% endfor %}"

# replace chat_template with default one
if tokenizer_config['chat_template'] != llamafact_default_chat_template:
    tokenizer_config['chat_template'] = llamafact_default_chat_template
    with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, ensure_ascii=False)
    print(f'chat_template in {tokenizer_config_path} has been replaced with default one')
else:
    print(f'chat_template in {tokenizer_config_path} is already the default one')
