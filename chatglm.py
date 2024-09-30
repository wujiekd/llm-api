import torch
import random
import numpy as np
import argparse
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
from lora import merge_to_lora_recursively,convert_to_lora_recursively
from model_chatglm_6b.src.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig
from model_chatglm_6b.src.tokenization_chatglm import ChatGLMTokenizer
import os

from streamers import TextIteratorStreamer
import yaml
from lora import LoraLinear,LoraLinear_merge
from threading import Thread
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

# 将config.yaml转换为Python对象
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class LLMPredictor():
    def __init__(self,device_ids,max_length=2048):
        self.set_random_seed(42)
        
        if len(device_ids) == 1:
            self.device_id = device_ids[0]
        # parms
        self.seq_len = max_length

           
    
    def loadmodel(self,cfg):
        if cfg.start_lora:
            config = ChatGLMConfig.from_json_file(os.path.join(cfg.config_path,'config.json'))
            self.tokenizer = ChatGLMTokenizer.from_pretrained(cfg.config_path) 
            self.model = ChatGLMForConditionalGeneration(config)
            convert_to_lora_recursively(self.model, lora_rank=8, lora_alpha=8, lora_dropout=0.1)
            self.model.load_state_dict(torch.load(cfg.model_path, map_location='cpu'), strict=True)
            
            self.model = self.model.half().cuda(torch.device('cuda:{}'.format(self.device_id)))
            self.model.eval()
            
            
        else:
            config = ChatGLMConfig.from_json_file(os.path.join(cfg.config_path,'config.json'))
            self.tokenizer = ChatGLMTokenizer.from_pretrained(cfg.config_path) 
            self.model = ChatGLMForConditionalGeneration(config)
            self.model.load_state_dict(torch.load(cfg.model_path, map_location='cpu'), strict=True)
            
            self.model = self.model.half().cuda(torch.device('cuda:{}'.format(self.device_id)))
            self.model.eval()
            
        
    # Fix seed and predict deterministically
    def set_random_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    def torch_gc(self):
        if torch.cuda.is_available():
            CUDA_DEVICE = f"cuda:{self.device_id}"
            with torch.cuda.device(CUDA_DEVICE):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


        
    def reading_comprehension(self , prompt ): # router = '不相关'
        self.set_random_seed(42)
        response =  self.model.chat(self.tokenizer,
                                    prompt,
                                    history=None,
                                    max_new_tokens=self.seq_len, do_sample=False)
        
        self.torch_gc()
        return response 
    
    
    
        
    def reading_comprehension_streamer(self  ,prompt ): # router = '不相关'
        self.set_random_seed(42)
        count=0
        response_temp=''
        for response in self.model.stream_chat(self.tokenizer,
                                    prompt,
                                    history=None,
                                    max_new_tokens=self.seq_len, do_sample=False):
            if response_temp==response:# stop by <pad>
                count+=1
                if count>50:
                    break
            else:
                response_temp=response
                count=0
                
            yield response
        self.torch_gc()
        
        
    # def batch_reading_comprehension(self , batch_question: List[str]= None): 
    #     self.set_random_seed(42)

    #     batch_response = self.model.batch_chat(self.tokenizer, batch_question,
    #                                 max_new_tokens=self.seq_len, do_sample=False)

    #     self.torch_gc()

        
    #     return batch_response

    
    # def batch_reading_comprehension_streamer(self , batch_question: List[str]= None): 
    #     self.set_random_seed(42)

    #     streamer = TextIteratorStreamer(self.tokenizer)
    #     for response in self.model.batch_chat_stream(self.tokenizer, streamer , batch_question
    #                                                  ,max_new_tokens=self.seq_len, do_sample=False):

    #         yield response

    #     self.torch_gc()
 
if __name__ == '__main__':
    config = './config/chatglm.yaml'
    # 加载config.yaml文件
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg = Config(**cfg)
 
    LLM = LLMPredictor([1])    
    LLM.loadmodel(cfg)
    
    texts = ['你晚上吃饭了吗','你晚上吃饭了吗', '你好，如何处理用户隐私问题？']   
    reponsr = LLM.batch_reading_comprehension(texts)
    print(reponsr)
    
    
    for reponsr in LLM.batch_reading_comprehension_streamer(texts):
        print(reponsr)
    
    print(LLM.reading_comprehension('你晚上吃饭了吗'))
