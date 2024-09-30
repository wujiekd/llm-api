import torch
import random
import numpy as np
import torch
from lora import merge_to_lora_recursively
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation import GenerationConfig

# from model_qwen.src.modeling_qwen import QWenLMHeadModel
# from model_qwen.src.configuration_qwen import QWenConfig
# from model_qwen.src.tokenization_qwen import QWenTokenizer
# from model_qwen.src.qwen_generation_utils import make_context
# from lora import convert_to_lora_recursively
from transformers.generation.configuration_utils import GenerationConfig

# from transformers import TextIteratorStreamer


import yaml
from streamers import TextIteratorStreamer

# 将config.yaml转换为Python对象
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

# system = '假设你是腾讯云产品的人工智能助手，请根据以上检索到的信息完成以下用户的提问。\n注意，如果检索信息和用户问题不符也请尝试回答。\n回答要先介绍相关背景知识然后再给出详细的解释和方法步骤。\n'
#system = '假设你是腾讯云产品的人工智能助手，你的名字叫Copilot，请根据以上检索到的信息完成以下用户的提问。\n注意，如果检索信息和用户问题不符也请尝试回答。\n回答要先介绍相关背景知识然后再给出详细的解释和方法步骤。\n'
  
class QWenLLMPredictor():
    def __init__(self, cfg, device_ids):
        self.set_random_seed(42)
        
   
        self.device_ids = device_ids
        self.cfg = cfg
        self.loadmodel(cfg)
        
    
    def loadmodel(self, cfg):        
        #self.tokenizer = QWenTokenizer.from_pretrained(cfg.config_path,padding_side="left", pad_token="<|endoftext|>") 
        model_path = cfg.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"Model directory: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", local_files_only=True, 
            trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)


        
        # use bf16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
        # use fp16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
        # use cpu only
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
        # use auto mode, automatically select precision based on the device.
       


        # if cfg.start_lora:        
        #     pth_path = cfg.model_path
        #     if not os.path.exists(pth_path):
        #         print('lora model not exists:', pth_path)
            
        #     print('create qw lora model ...')
        #     convert_to_lora_recursively(self.model, lora_rank=8, lora_alpha=8, lora_dropout=0.1)
        #     print('convert to lora model')
            
        #     print('load lora model:', pth_path)
        #     result = self.model.load_state_dict(
        #         torch.load(pth_path, map_location='cpu'), strict=False)
        #     print('load result:', result)

        # self.model = self.model.half()
        self.model = self.model.eval()

        print("Current allocated memory:", torch.cuda.memory_allocated())
        
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
        
        
    def reading_comprehension(self , prompt): 
        
        # PROMPT = system
        # if len(prompt) > 0:
        #     context = prompt.replace('请用下文信息推理来回答下面问题，如果问题与后文无关，请回答不相关\n文本:\n', '')
        #     splits = context.split('\n问题:\n')
        #     if len(splits) != 2:
        #         context = PROMPT
        #         question = prompt
        #     else:
        #         context = splits[0] + '\n' + PROMPT
        #         question = splits[1]
        # else:
        #     context = PROMPT
        #     question = prompt
            
        # self.set_random_seed(42)
        # response, history =  self.model.chat(self.tokenizer,
        #                                     question,
        #                             history=None,
        #                             system = context
        #                            )
        

        # self.multi_gpu_torch_gc()

        self.set_random_seed(42)
        response, history =  self.model.chat(self.tokenizer,
                                            prompt,
                                    history=None,
                                   )
        

        self.multi_gpu_torch_gc()

        return response, history
    
        
    # def reading_comprehension_streamer(self ,prompt ): 
        
    #     # PROMPT = system
        
        
    #     # if len(prompt) > 0:
    #     #     context = prompt.replace('请用下文信息推理来回答下面问题，如果问题与后文无关，请回答不相关\n文本:\n', '')
    #     #     splits = context.split('\n问题:\n')
    #     #     if len(splits) != 2:
    #     #         context = PROMPT
    #     #         question = prompt
    #     #     else:
    #     #         context = splits[0] + '\n' + PROMPT
    #     #         question = splits[1]
    #     # else:
    #     #     context = PROMPT
    #     #     question = prompt
            
    #     # self.set_random_seed(42)
    #     # count=0
    #     # response_temp=''
        
    #     # streamer = TextIteratorStreamer(self.tokenizer,skip_prompt=True, decode_kwargs  = dict(skip_special_tokens=True, errors='ignore'))
    #     # for response in self.model.stream_chat(streamer,
    #     #                                        self.tokenizer,
    #     #                                     question,
    #     #                             history=None,
    #     #                             system = context
    #     #                            ):
    #     #     if response_temp==response:# stop by <pad>
    #     #         count+=1
    #     #         if count>50:
    #     #             break
    #     #     else:
    #     #         response_temp=response
    #     #         count=0
                
    #     #     yield response
    #     self.multi_gpu_torch_gc()

 
    
    
    # def batch_reading_comprehension(self , batch_question): 
        
    #     # self.set_random_seed(42)
        
        
    #     # batch_question = [f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{i}<|im_end|>\n<|im_start|>assistant\n" for i in batch_question]
        
    #     # batch_response = self.model.batch_chat(self.tokenizer, batch_question)

    #     self.multi_gpu_torch_gc()
        

    #     # return batch_response
    
    # def batch_reading_comprehension_streamer(self , batch_question): 
        
        
    #     # PROMPT = system
        
    #     # self.set_random_seed(42)
        
    #     # for i in range(len(batch_question)):
    #     #     prompt = batch_question[i]
    #     #     if len(prompt) > 0:
    #     #         context = prompt.replace('请用下文信息推理来回答下面问题，如果问题与后文无关，请回答不相关\n文本:\n', '')
    #     #         splits = context.split('\n问题:\n')
    #     #         if len(splits) != 2:
    #     #             context = PROMPT
    #     #             question = prompt
    #     #         else:
    #     #             context = splits[0] + '\n' + PROMPT
    #     #             question = splits[1]
    #     #     else:
    #     #         context = PROMPT
    #     #         question = prompt
                
            
    #     #     batch_question[i] = f"<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            
        
        
    #     # streamer = TextIteratorStreamer(self.tokenizer,skip_prompt=True, decode_kwargs  = dict(skip_special_tokens=True, errors='ignore'))
    #     # for response in self.model.batch_chat_stream(self.tokenizer, streamer , batch_question):
            
    #     #     yield response

    #     # self.multi_gpu_torch_gc()


        
if __name__ == '__main__':
    config = './config/qwen.yaml'
    # 加载config.yaml文件
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg = Config(**cfg)
    
    device_id = [0]
    LLM = QWenLLMPredictor(device_id)    
    LLM.loadmodel(cfg)
    
    prompt = '如何处理用户隐私问题？'
    
    path = '/apdcephfs_share_887471/share_887471/staffs/kedalu/our_exp/bigmodel/scripts2/t_chat_record14_1_6.json'
    
    f = open(path + '_res_llm.jsonl', 'w')
    for line in open(path):
        line = line.strip()
        input = json.loads(line)
        question = input['Prompt']
        
        for response in LLM.reading_comprehension_streamer(question):
            print(response)
        
        content= {}
        # content['question'] = input['llm_input']
        content['response'] = response
        f.write(json.dumps(content, ensure_ascii=False) + '\n')
        f.flush()
        #response = requests.post(url, headers=headers, data=json.dumps(data))
            # print (response)
        
            # content = json.loads(response)
            # if content['final'] == True:
            #     content['question'] = question
            #     if 'gt' in input.keys():
            #         content['gt'] = input['gt']
                # f.write(json.dumps(content, ensure_ascii=False) + '\n')
                # f.flush()
            # else:
            #     print(content['response'])
    f.close()
        
    
        
    # answer = LLM.reading_comprehension(prompt,[])
    # print(answer)
    
    
    # batch_question = ["今天我想吃点啥，甜甜的，推荐下",
    #             "我马上迟到了，怎么做才能不迟到",
    #             "如何处理用户隐私问题？",
    #             "如何处理用户隐私问题？",
    #             ]
    
    # for response in LLM.batch_reading_comprehension_streamer(batch_question):
    #     break
    #     #print(response)






{"ID":306314,"UserID":46153,"SessionID":"923eb249-7ef8-42f8-9271-76368a5d4c31","Score":0,"Reason":"","Question":"如何查看定时任务","QuestionReplaced":"","Answer":"","StaffName":"70-p_qxgeng",
 "Prompt":"请用下文信息推理来回答下面问题，如果问题与后文无关，请回答不相关\n文本:\n日志服务_查看任务 ## 操作场景\n\n本文为您介绍如何查看定时 SQL 分析任务信息。\n操作步骤\n\n1.登录 [日志服务控制台](https://console.cloud.tencent.com/cls/overview \"https://console.cloud.tencent.com/cls/overview\")。\n\n2.在左侧导航栏中，选择**数据处理 \\\u003e 定时 SQL 分析**，即可查看如下信息。\n\n任务的基本信息：包括任务的名称、任务的 ID、源日志主题、目标日志主题、任务创建时间、最近修改时间、调度周期、调度范围、SQL 时间窗口。\n\n弹性伸缩_查询定时任务 ## 1\\. 接口描述\n\n接口请求域名： as.tencentcloudapi.com 。\n\n本接口 (DescribeScheduledActions) 用于查询一个或多个定时任务的详细信息。\n\n- 可以根据定时任务ID、定时任务名称或者伸缩组ID等信息来查询定时任务的详细信息。过滤信息详细请见过滤器 `Filter`。\n- 如果参数为空，返回当前用户一定数量（Limit所指定的数量，默认为20）的定时任务。\n\n默认接口请求频率限制：20次/秒。\n默认接口请求频率限制：20次/秒。\n\n推荐使用 API Explorer\n\n\n[点击调试](https://console.cloud.tencent.com/api/explorer?Product=as\u0026Version=2018-04-19\u0026Action=DescribeScheduledActions)\n\nAPI Explorer 提供了在线调用、签名验证、SDK 代码生成和快速检索接口等能力。您可查看每次调用的请求内容和返回结果以及自动生成 SDK 调用示例。\n\n云开发后端定时任务如何实现 您好，您可以使用 [定时触发器](https://cloud.tencent.com/document/product/876/32314) 结合云函数，实现定时任务的功能。\n**温馨提示：**\n您好，推荐您前往 [云开发社区](https://developers.weixin.qq.com/community/minihome/mixflow/1286298401038155776) 进行提问或交流，社区有专人值班回复您的问题。同时开发者社区有更多的学习资料和开发文档可以帮助您高效开发。\n\n弹性伸缩_查询定时任务 | 参数名称 | 必选 | 类型 | 描述 |\n| --- | --- | --- | --- |\n| Action | 是 | String | [公共参数](/document/api/377/20426)，本接口取值：DescribeScheduledActions。 |\n| Version | 是 | String | [公共参数](/document/api/377/20426)，本接口取值：2018-04-19。 |\n| Region | 是 | String | [公共参数](/document/api/377/20426)，详见产品支持的 [地域列表](/document/api/377/20426)。 |\n| ScheduledActionIds.N | 否 | Array of String | 按照一个或者多个定时任务ID查询。实例ID形如：asst-am691zxo。每次请求的实例的上限为100。参数不支持同时指定ScheduledActionIds和Filters。\u003cbr\u003e示例值：\\[\"asst-caa5ha40\"\\] |\n| Filters.N | 否 | Array of [Filter](/document/api/377/20453#Filter) | 过滤条件。\u003cbr\u003escheduled-action-id - String - 是否必填：否 -（过滤条件）按照定时任务ID过滤。\u003cbr\u003escheduled-action-name - String - 是否必填：否 - （过滤条件） 按照定时任务名称过滤。\u003cbr\u003eauto-scaling-group-id - String - 是否必填：否 - （过滤条件） 按照伸缩组ID过滤。 |\n\n弹性伸缩_查询定时任务 3\\. 输出参数\n\n| 参数名称 | 类型 | 描述 |\n| --- | --- | --- |\n| TotalCount | Integer | 符合条件的定时任务数量。\u003cbr\u003e示例值：1 |\n| ScheduledActionSet | Array of [ScheduledAction](/document/api/377/20453#ScheduledAction) | 定时任务详细信息列表。 |\n| RequestId | String | 唯一请求 ID，每次请求都会返回。定位问题时需要提供该次请求的 RequestId。 |\n示例1 查询定时任务\n\n#### 输入示例\n\n```\nPOST / HTTP/1.1\nHost: as.tencentcloudapi.com\nContent-Type: application/json\nX-TC-Action: DescribeScheduledActions\n\u003c公共请求参数\u003e\n\n{\n    \"ScheduledActionIds\": [\n        \"asst-caa5ha40\"\n    ]\n}\n```\n输出示例\n问题:\n如何查看定时任务",
 "IndexContent":"{\"contents\":null,\"sources\":[\"andon\",\"andon\",\"andon\",\"andon\",\"andon\"],\"ids\":[\"78893\",\"20450\",\"31647\",\"20450\",\"20450\"],\"links\":[\"https://cloud.tencent.com/document/product/614/78893\",\"https://cloud.tencent.com/document/product/377/20450\",\"\",\"https://cloud.tencent.com/document/product/377/20450\",\"https://cloud.tencent.com/document/product/377/20450\"],\"texts\":[\"日志服务_查看任务\",\"弹性伸缩_查询定时任务\",\"云开发后端定时任务如何实现\",\"弹性伸缩_查询定时任务\",\"弹性伸缩_查询定时任务\"],\"scores\":[1.1977425,0.9326108,0.7553064,0.6441555,0.55900866],\"source_types\":[\"andon\",\"andon\",\"andon\",\"andon\",\"andon\"],\"doc_types\":[\"doc\",\"doc\",\"qa\",\"doc\",\"doc\"],\"page_contents\":[\"## 操作场景\\n\\n本文为您介绍如何查看定时 SQL 分析任务信息。\\n操作步骤\\n\\n1.登录 [日志服务控制台](https://console.cloud.tencent.com/cls/overview \\\"https://console.cloud.tencent.com/cls/overview\\\")。\\n\\n2.在左侧导航栏中，选择**数据处理 \\\\\\u003e 定时 SQL 分析**，即可查看如下信息。\\n\\n任务的基本信息：包括任务的名称、任务的 ID、源日志主题、目标日志主题、任务创建时间、最近修改时间、调度周期、调度范围、SQL 时间窗口。\",\"## 1\\\\. 接口描述\\n\\n接口请求域名： as.tencentcloudapi.com 。\\n\\n本接口 (DescribeScheduledActions) 用于查询一个或多个定时任务的详细信息。\\n\\n- 可以根据定时任务ID、定时任务名称或者伸缩组ID等信息来查询定时任务的详细信息。过滤信息详细请见过滤器 `Filter`。\\n- 如果参数为空，返回当前用户一定数量（Limit所指定的数量，默认为20）的定时任务。\\n\\n默认接口请求频率限制：20次/秒。\\n默认接口请求频率限制：20次/秒。\\n\\n推荐使用 API Explorer\\n\\n\\n[点击调试](https://console.cloud.tencent.com/api/explorer?Product=as\\u0026Version=2018-04-19\\u0026Action=DescribeScheduledActions)\\n\\nAPI Explorer 提供了在线调用、签名验证、SDK 代码生成和快速检索接口等能力。您可查看每次调用的请求内容和返回结果以及自动生成 SDK 调用示例。\",\"您好，您可以使用 [定时触发器](https://cloud.tencent.com/document/product/876/32314) 结合云函数，实现定时任务的功能。\\n**温馨提示：**\\n您好，推荐您前往 [云开发社区](https://developers.weixin.qq.com/community/minihome/mixflow/1286298401038155776) 进行提问或交流，社区有专人值班回复您的问题。同时开发者社区有更多的学习资料和开发文档可以帮助您高效开发。\",\"| 参数名称 | 必选 | 类型 | 描述 |\\n| --- | --- | --- | --- |\\n| Action | 是 | String | [公共参数](/document/api/377/20426)，本接口取值：DescribeScheduledActions。 |\\n| Version | 是 | String | [公共参数](/document/api/377/20426)，本接口取值：2018-04-19。 |\\n| Region | 是 | String | [公共参数](/document/api/377/20426)，详见产品支持的 [地域列表](/document/api/377/20426#.E5.9C.B0.E5.9F.9F.E5.88.97.E8.A1.A8)。 |\\n| ScheduledActionIds.N | 否 | Array of String | 按照一个或者多个定时任务ID查询。实例ID形如：asst-am691zxo。每次请求的实例的上限为100。参数不支持同时指定ScheduledActionIds和Filters。\\u003cbr\\u003e示例值：\\\\[\\\"asst-caa5ha40\\\"\\\\] |\\n| Filters.N | 否 | Array of [Filter](/document/api/377/20453#Filter) | 过滤条件。\\u003cbr\\u003escheduled-action-id - String - 是否必填：否 -（过滤条件）按照定时任务ID过滤。\\u003cbr\\u003escheduled-action-name - String - 是否必填：否 - （过滤条件） 按照定时任务名称过滤。\\u003cbr\\u003eauto-scaling-group-id - String - 是否必填：否 - （过滤条件） 按照伸缩组ID过滤。 |\",\"3\\\\. 输出参数\\n\\n| 参数名称 | 类型 | 描述 |\\n| --- | --- | --- |\\n| TotalCount | Integer | 符合条件的定时任务数量。\\u003cbr\\u003e示例值：1 |\\n| ScheduledActionSet | Array of [ScheduledAction](/document/api/377/20453#ScheduledAction) | 定时任务详细信息列表。 |\\n| RequestId | String | 唯一请求 ID，每次请求都会返回。定位问题时需要提供该次请求的 RequestId。 |\\n示例1 查询定时任务\\n\\n#### 输入示例\\n\\n```\\nPOST / HTTP/1.1\\nHost: as.tencentcloudapi.com\\nContent-Type: application/json\\nX-TC-Action: DescribeScheduledActions\\n\\u003c公共请求参数\\u003e\\n\\n{\\n    \\\"ScheduledActionIds\\\": [\\n        \\\"asst-caa5ha40\\\"\\n    ]\\n}\\n```\\n输出示例\"],\"org_datas\":[\"## 操作场景\\n\\n本文为您介绍如何查看定时 SQL 分析任务信息。\\n操作步骤\\n\\n1.登录 [日志服务控制台](https://console.cloud.tencent.com/cls/overview \\\"https://console.cloud.tencent.com/cls/overview\\\")。\\n\\n2.在左侧导航栏中，选择**数据处理 \\\\\\u003e 定时 SQL 分析**，即可查看如下信息。\\n\\n任务的基本信息：包括任务的名称、任务的 ID、源日志主题、目标日志主题、任务创建时间、最近修改时间、调度周期、调度范围、SQL 时间窗口。\",\"## 1\\\\. 接口描述\\n\\n接口请求域名： as.tencentcloudapi.com 。\\n\\n本接口 (DescribeScheduledActions) 用于查询一个或多个定时任务的详细信息。\\n\\n- 可以根据定时任务ID、定时任务名称或者伸缩组ID等信息来查询定时任务的详细信息。过滤信息详细请见过滤器 `Filter`。\\n- 如果参数为空，返回当前用户一定数量（Limit所指定的数量，默认为20）的定时任务。\\n\\n默认接口请求频率限制：20次/秒。\\n默认接口请求频率限制：20次/秒。\\n\\n推荐使用 API Explorer\\n\\n\\n[点击调试](https://console.cloud.tencent.com/api/explorer?Product=as\\u0026Version=2018-04-19\\u0026Action=DescribeScheduledActions)\\n\\nAPI Explorer 提供了在线调用、签名验证、SDK 代码生成和快速检索接口等能力。您可查看每次调用的请求内容和返回结果以及自动生成 SDK 调用示例。\",\"您好，您可以使用 [定时触发器](https://cloud.tencent.com/document/product/876/32314) 结合云函数，实现定时任务的功能。\\n**温馨提示：**\\n您好，推荐您前往 [云开发社区](https://developers.weixin.qq.com/community/minihome/mixflow/1286298401038155776) 进行提问或交流，社区有专人值班回复您的问题。同时开发者社区有更多的学习资料和开发文档可以帮助您高效开发。\",\"| 参数名称 | 必选 | 类型 | 描述 |\\n| --- | --- | --- | --- |\\n| Action | 是 | String | [公共参数](/document/api/377/20426)，本接口取值：DescribeScheduledActions。 |\\n| Version | 是 | String | [公共参数](/document/api/377/20426)，本接口取值：2018-04-19。 |\\n| Region | 是 | String | [公共参数](/document/api/377/20426)，详见产品支持的 [地域列表](/document/api/377/20426)。 |\\n| ScheduledActionIds.N | 否 | Array of String | 按照一个或者多个定时任务ID查询。实例ID形如：asst-am691zxo。每次请求的实例的上限为100。参数不支持同时指定ScheduledActionIds和Filters。\\u003cbr\\u003e示例值：\\\\[\\\"asst-caa5ha40\\\"\\\\] |\\n| Filters.N | 否 | Array of [Filter](/document/api/377/20453#Filter) | 过滤条件。\\u003cbr\\u003escheduled-action-id - String - 是否必填：否 -（过滤条件）按照定时任务ID过滤。\\u003cbr\\u003escheduled-action-name - String - 是否必填：否 - （过滤条件） 按照定时任务名称过滤。\\u003cbr\\u003eauto-scaling-group-id - String - 是否必填：否 - （过滤条件） 按照伸缩组ID过滤。 |\",\"3\\\\. 输出参数\\n\\n| 参数名称 | 类型 | 描述 |\\n| --- | --- | --- |\\n| TotalCount | Integer | 符合条件的定时任务数量。\\u003cbr\\u003e示例值：1 |\\n| ScheduledActionSet | Array of [ScheduledAction](/document/api/377/20453#ScheduledAction) | 定时任务详细信息列表。 |\\n| RequestId | String | 唯一请求 ID，每次请求都会返回。定位问题时需要提供该次请求的 RequestId。 |\\n示例1 查询定时任务\\n\\n#### 输入示例\\n\\n```\\nPOST / HTTP/1.1\\nHost: as.tencentcloudapi.com\\nContent-Type: application/json\\nX-TC-Action: DescribeScheduledActions\\n\\u003c公共请求参数\\u003e\\n\\n{\\n    \\\"ScheduledActionIds\\\": [\\n        \\\"asst-caa5ha40\\\"\\n    ]\\n}\\n```\\n输出示例\"],\"org_titles\":[\"日志服务_查看任务\",\"弹性伸缩_查询定时任务\",\"云开发后端定时任务如何实现\",\"弹性伸缩_查询定时任务\",\"弹性伸缩_查询定时任务\"],\"products\":[\"\",\"\",\"\",\"\",\"\"]}","ModelID":14,"TraceID":"b0f9d93e-6d21-41f5-b4b4-e37620975b8b","CreateTime":"2023-08-30T03:44:24+08:00","UpdateTime":"0001-01-01T00:00:00Z"}
