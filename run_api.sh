#!/bin/bash

# source venv/bin/activate

#cfg_path=./config/baichuan2-7b-chat.yaml
#cfg_path=./config/baichuan2-13b-chat.yaml
cfg_path=./config/qwen-14b-chat.yaml
#cfg_path=./config/qwen-7b-chat.yaml
port=8081

python3 api_stream_llm.py --config $cfg_path --port $port
