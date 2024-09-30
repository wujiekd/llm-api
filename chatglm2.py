import torch
import random
import numpy as np
import torch
from lora import merge_to_lora_recursively
from model_chatglm2_6b.src.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig
from model_chatglm2_6b.src.tokenization_chatglm import ChatGLMTokenizer
import os


class LLMPredictor2():
    def __init__(self, device_ids,max_length=4096,top_p =0.7,temperature=0.95):
        self.set_random_seed(42)
        
        if len(device_ids) == 1:
            self.device_id = device_ids[0]
        # parms
        self.max_length = max_length
        self.top_p = top_p
        self.temperature = temperature
        
        
    
    def loadmodel(self,cfg):
        
         # load model
        config = ChatGLMConfig.from_json_file(os.path.join(cfg.config_path,'config.json'))
        self.tokenizer = ChatGLMTokenizer.from_pretrained(cfg.config_path) 
        self.model = ChatGLMForConditionalGeneration(config)
        # merge_to_lora_recursively(self.model, lora_rank=8, lora_alpha=8, lora_dropout=0.1)
        self.model.load_state_dict(torch.load(cfg.model_path, map_location='cpu'), strict=False)
        
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

        
    # def refine_question(self , question: str,history : list[list]):  #'怎么吃饭？',[['你早上吃饭了没？','吃了'],['你中午吃饭了没？','吃了'],['你晚上吃饭了没？','吃了']]
        
    #     if history == None or history ==[]:
    #         return question
    
    #     prompt = f'根据以下问答内容，理解问题{len(history)+1}的意图，并把问题{len(history)+1}以完整的形式输出：\n'
    #     for i in range(len(history)):
    #         prompt+=f"问题{i+1}：{history[i][0]}\n答案{i+1}：{history[i][1]}\n"
        
    #     prompt+=f"问题{len(history)+1}：{question}"

    #     response, history = self.model.chat(self.tokenizer,
    #                             prompt,
    #                             history=history,
    #                             max_length = self.max_length,
    #                             top_p = self.top_p,
    #                             temperature = self.temperature ,
    #                             ori_query=question,
    #                             infer_with_lora=True,
    #                             lora_mode='refine')

    #     self.torch_gc()
    #     return response
    
    # def relevant_route(self, prompt: str,history: list[list]):  
        
    #     response, history =  self.model.chat(self.tokenizer,
    #                                 prompt,
    #                                 history=history,
    #                                 max_length = self.max_length,
    #                                 top_p = self.top_p,
    #                                 temperature = self.temperature ,
    #                                 infer_with_lora=True,
    #                                 lora_mode='route')
        
    #     self.torch_gc()
    #     return response
    
        
    # def closed_question(self, prompt: str,ori_prompt: str,history: list[list]): 

    #     response, history =  self.model.chat(self.tokenizer,
    #                                 prompt,
    #                                 history=history,
    #                                 max_length=self.max_length,
    #                                 top_p=self.top_p,
    #                                 temperature=self.temperature ,
    #                                 infer_with_lora=True,
    #                                 ori_query=ori_prompt,
    #                                 lora_mode='read')
        
    #     self.torch_gc()
    #     return response 

    # def closed_question_streamer(self, prompt: str,ori_prompt: str,history: list[list]):  

        
    #     count=0
    #     response_temp=''
    #     for response, history in self.model.stream_chat(self.tokenizer,
    #                                 prompt,
    #                                 history=history,
    #                                 max_length=self.max_length,
    #                                 top_p=self.top_p,
    #                                 temperature=self.temperature ,
    #                                 infer_with_lora=True,
    #                                 ori_query=ori_prompt,
    #                                 lora_mode='read'):
    #         if response_temp==response:# stop by <pad>
    #             count+=1
    #             if count>50:
    #                 break
    #         else:
    #             response_temp=response
    #             count=0
                
    #         yield response
        
    #     self.torch_gc()


    def open_question(self, prompt,history):  
        
        response, history =  self.model.chat(self.tokenizer,
                                    prompt,
                                    history=history,
                                    max_length=self.max_length,
                                    top_p=self.top_p,
                                    temperature=self.temperature ,
                                   )
        
        self.torch_gc()
        return response 
        
    def open_question_streamer(self, prompt,history):  
        
        
        count=0
        response_temp=''
        for response, history in self.model.stream_chat(self.tokenizer,
                                    prompt,
                                    history=history,
                                    max_length=self.max_length,
                                    top_p=self.top_p,
                                    temperature=self.temperature ,
                                    ):
            if response_temp==response:# stop by <pad>
                count+=1
                if count>50:
                    break
            else:
                response_temp=response
                count=0
                
            yield response
        self.torch_gc()
        
    def reading_comprehension(self , prompt ,history): 
        

        
        
        # quest_index = prompt.rfind('问题:')
        
        # if quest_index!=-1:
        #     prompt = prompt[quest_index+3:]
        #     prompt = '假如你是腾讯云客服助手，请结合腾讯云产品信息并用中文回答以下问题\n问：{question}\n答：'.format(question=prompt)
    
        
        response  = self.open_question(prompt,history)
        
        self.torch_gc()
        return response 
    
    
    
        
    def reading_comprehension_streamer(self  ,prompt ,history): 
        
        # quest_index = prompt.rfind('问题:')
        
        # if quest_index!=-1:
        #     prompt = prompt[quest_index+3:]
        #     prompt = '假如你是腾讯云客服助手，请结合腾讯云产品信息并用中文回答以下问题\n问：{question}\n答：'.format(question=prompt)
    
    
        for response in self.open_question_streamer(prompt,history):
            yield response
            
        
        self.torch_gc()