#!/usr/bin/env python
# coding=utf-8

from lora import convert_to_lora_recursively, merge_lora
# from model_chatglm_6b_new.src.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BloomTokenizerFast,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    AutoModel,
)
import torch
import os
import sys

def main():
    # model_path = sys.argv[1]
    # config = ChatGLMConfig.from_json_file(os.path.join(model_path, "config.json"))
    # model = ChatGLMForConditionalGeneration(config).half().cuda()
    # convert_to_lora_recursively(model, lora_rank=8, lora_alpha=8, lora_dropout=0.1)
    # model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu'), strict=True)
    # merge_lora(model)
    # for name, param in model.named_parameters():
    #     print(name, param.dtype)
    # torch.save(model.state_dict(), 'model_chatglm_6b_new/model/pytorch_model.bin')
    
    
    model_path = '/apdcephfs_share_887471/share_887471/projects/doclm/service/models/tencent_cloud/v1.6.5/tcloud_rel/tcloud_rel.bin'
    config_path = 'model_bloomz/pretrain'
    config = AutoConfig.from_pretrained(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config_path, use_fast=True)
    model = AutoModelForCausalLM.from_config(config)
    convert_to_lora_recursively(model, lora_rank=8, lora_alpha=8, lora_dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    merge_lora(model)
    
    for name, param in model.named_parameters():
        print(name, param.dtype)
    torch.save(model.state_dict(), '/apdcephfs_share_887471/share_887471/projects/doclm/service/models/tencent_cloud/v1.6.5/tcloud_rel/merge_tcloud_rel.bin')
            

if __name__ == "__main__":
    main()