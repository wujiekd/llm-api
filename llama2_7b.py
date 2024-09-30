import torch
import random
import numpy as np
import torch
from lora import merge_to_lora_recursively
import os
from transformers import AutoTokenizer, LlamaForCausalLM
        
        
class LLMPredictor6():
    def __init__(self, device_ids):
        self.set_random_seed(42)
        self.device_ids = device_ids
    
    def loadmodel(self,cfg):
        model_path = cfg.model_path
        
        self.model = LlamaForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)



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
        

    def open_question(self, prompt,history):  
        self.set_random_seed(42)
        # response, history =  self.model.chat(self.tokenizer,
        #                                     prompt,
        #                             history=history
        #                            )

        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate
        generate_ids = self.model.generate(inputs.input_ids, max_length=4096)
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        self.multi_gpu_torch_gc()
        return response 
        

        
    def reading_comprehension(self , prompt ,history): 
        response  = self.open_question(prompt,history)
        
        return response 
    
    
    
