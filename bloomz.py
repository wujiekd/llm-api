import torch
import random
import numpy as np
import torch
import json
from lora import merge_to_lora_recursively,convert_to_lora_recursively,merge_lora

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

from streamers import TextIteratorStreamer
import yaml
from lora import LoraLinear,LoraLinear_merge
from threading import Thread


# 将config.yaml转换为Python对象
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
def set_random_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        

def chat(model, tokenizer, query,device="cuda:0"):
  
    question = query.strip('\n').strip()
                
    text = "user:" + question + "\n" + "bot:"
        
    torch.manual_seed(0)

    # inputs = tokenizer.encode(text, return_tensors="pt")
    inputs = tokenizer([text], return_tensors="pt")
    inputs = inputs.to(device)
    
    generation_kwargs = dict(do_sample=True, top_p=0.8,
                                     num_return_sequences=1, max_new_tokens=4096)
    outputs = model.generate(inputs.input_ids,
                            attention_mask = inputs.attention_mask,
                **generation_kwargs)
    
    generated_text = tokenizer.decode(outputs.cpu().numpy()[0][inputs.input_ids.shape[1]:])
    generated_text = generated_text.replace("</s>", "")

    return generated_text

def stream_chat(model, tokenizer,streamer, query,device="cuda:0"):

    question = query.strip('\n').strip()
                
    text = "user:" + question + "\n" + "bot:"
    
    torch.manual_seed(0)
    
    texts = [text]
    inputs = tokenizer(texts, return_tensors="pt")
    inputs = inputs.to(device)
        
        
    generation_kwargs = dict(inputs, streamer=streamer, do_sample=True, top_p=0.8,
                                         num_return_sequences=1, max_new_tokens=4096,attention_mask = inputs.attention_mask)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    batchsize = len(texts)
    generated_text = ''
    
    
    for new_text in streamer:
        for i in range(batchsize):
            generated_text +=new_text[0]
            
        generated_text = generated_text.replace("<pad>", "")
        generated_text = generated_text.replace("</s>", "")    
        yield generated_text

                
class LLMPredictor3():
    def __init__(self,device_ids,max_length=4096,top_p =0.8):
        self.set_random_seed(42)
        
        if len(device_ids) == 1:
            self.device_id = device_ids[0]
   


        
    def loadmodel(self,cfg):
        
        if cfg.start_lora:
            self.config = AutoConfig.from_pretrained(cfg.config_path)
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.config_path, use_fast=True)
            self.model = AutoModelForCausalLM.from_config(self.config)
            convert_to_lora_recursively(self.model, lora_rank=8, lora_alpha=8, lora_dropout=0.1)
            self.model.load_state_dict(torch.load(cfg.model_path, map_location='cpu'), strict=True)
        
            

            self.model = self.model.half().cuda(torch.device('cuda:{}'.format(self.device_id)))
            merge_lora(self.model)
        else:
            self.config = AutoConfig.from_pretrained(cfg.config_path)
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.config_path, use_fast=True)
            self.model = AutoModelForCausalLM.from_config(self.config)
            self.model.load_state_dict(torch.load(cfg.model_path, map_location='cpu'), strict=True)   
            self.model = self.model.half().cuda(torch.device('cuda:{}'.format(self.device_id)))
            

        
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

    def reading_comprehension(self , prompt ): 
        response =  chat(self.model,
                                    self.tokenizer,
                                    prompt,
                                    device = 'cuda:{}'.format(self.device_id),)
        
        self.torch_gc()
        

        return response 
    
        
    def reading_comprehension_streamer(self ,prompt): 
        
        count=0
        response_temp=''
        streamer = TextIteratorStreamer(self.tokenizer,skip_prompt=True)
        for response in stream_chat(self.model,
                                             self.tokenizer,
                                             streamer,
                                    prompt,
                                    device = 'cuda:{}'.format(self.device_id)):
            if response_temp==response:# stop by <pad>
                count+=1
                if count>50:
                    break
            else:
                response_temp=response
                count=0
                
            yield response
        self.torch_gc()
        
    
    # def batch_reading_comprehension(self , batch_question): 
    #     self.set_random_seed(42)
        
    #     texts = [text.strip('\n').strip() for text in batch_question]
    #     texts = ["user:" + text + "\n" + "bot:" for text in texts]
            
    #     inputs = self.tokenizer(texts,padding=True, return_tensors="pt")
    #     inputs = inputs.to('cuda:{}'.format(self.device_id))
        

    #     generation_kwargs = dict(do_sample=True, top_p=0.8,
    #                                      num_return_sequences=1, max_new_tokens=4096)
    #     outputs = self.model.generate(inputs.input_ids,
    #                             attention_mask = inputs.attention_mask,
    #                 **generation_kwargs)
        
    #     pred_full_list = []
    #     for i in range(len(texts)):
    #         pred_full = self.tokenizer.decode(outputs.cpu().numpy()[i][inputs.input_ids.shape[1]:])
    #         pred_full = pred_full.replace("<pad>", "")
    #         pred_full = pred_full.replace("</s>", "")
    #         pred_full_list.append(pred_full)
        
    #     self.torch_gc()
        
    #     return pred_full_list
         
    
    # def batch_reading_comprehension_streamer(self , batch_question):  
    #     self.set_random_seed(42)
        
    #     texts = [text.strip('\n').strip() for text in batch_question]
    #     texts = ["user:" + text + "\n" + "bot:" for text in texts]
            
    #     inputs = self.tokenizer(texts,padding=True, return_tensors="pt")
        
    #     inputs = inputs.to('cuda:{}'.format(self.device_id))
        
    #     streamer = TextIteratorStreamer(self.tokenizer,skip_prompt=True)
    #     generation_kwargs = dict(inputs, streamer=streamer, do_sample=True, top_p=0.8,
    #                                      num_return_sequences=1, max_new_tokens=4096,attention_mask = inputs.attention_mask)
    #     thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
    #     thread.start()
    #     batchsize = len(texts)
    #     generated_text = ['' for _ in range(batchsize)]
        
    #     for new_text in streamer:
    #         pred_full_list = []
    #         for i in range(batchsize):
    #             generated_text[i] +=new_text[i]
    #             generated_text[i] = generated_text[i].replace("<pad>", "")
    #             generated_text[i] = generated_text[i].replace("</s>", "")
            
    #             pred_full_list.append(generated_text[i])

    #         yield pred_full_list
            
    #     self.torch_gc()
        
    
         
         
if __name__ == '__main__':
    config = './config/bloomz.yaml'
    # 加载config.yaml文件
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg = Config(**cfg)
 
    LLM = LLMPredictor3([0])    
    LLM.loadmodel(cfg)
    
    texts = ['你晚上吃饭了吗', '你晚上吃饭了吗', '你好，如何处理用户隐私问题？']   
    reponsr = LLM.batch_reading_comprehension(texts)
