#!/bin/bash

source venv/bin/activate

cfg_path=./config/prompt_post_loan.yaml
url="http://0.0.0.0:8081/chat"
input_fn='test_data/input.jsonl'
output_fn='./output.jsonl'

python3 get_data.py \
  --config $cfg_path \
  --url $url \
  --input_fn $input_fn \
  --output_fn $output_fn || exit 1

echo "Done! $(date)"
