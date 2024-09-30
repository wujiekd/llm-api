import torch
import random
import numpy as np
import torch
from lora import merge_to_lora_recursively
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


class BaichuanLLMPredictor():
    def __init__(self, cfg, device_ids, random_seed=42):
        """
        Initialization.
        """
        self.random_seed = random_seed
        self.set_random_seed(random_seed)
        self.device_ids = device_ids
        self.cfg = cfg
        self.loadmodel(cfg)

    def loadmodel(self, cfg):
        model_path = cfg.model_path
        print(f"Model directory: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map='auto', local_files_only=True, 
            trust_remote_code=True
        )
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.eval()

    # Fix seed and predict deterministically
    def set_random_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    def multi_gpu_torch_gc(self):
        if torch.cuda.is_available():
            for device_id in self.device_ids:
                CUDA_DEVICE = f"cuda:{device_id}"
                with torch.cuda.device(CUDA_DEVICE):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

    def open_question(self, prompt):  
        self.set_random_seed(self.random_seed)
        
        # inputs = self.tokenizer(prompt, return_tensors='pt')
        # inputs = inputs.to('cuda')
        # pred = self.model.generate(**inputs, max_new_tokens=256,repetition_penalty=1.1)
        # response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        
        # self.multi_gpu_torch_gc()
        # return response[len(prompt):]

        messages = [{"role": "user", "content": prompt}]
        response = self.model.chat(self.tokenizer, messages)
        self.multi_gpu_torch_gc()
        return response, None
        
    def reading_comprehension(self, prompt): 
        response, history  = self.open_question(prompt)
        return response, history
    
    
    
