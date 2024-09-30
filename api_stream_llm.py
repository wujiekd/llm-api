import argparse
import os
import torch

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn, json, datetime
# from chatglm import LLMPredictor
# from chatglm2 import LLMPredictor2
# from bloomz import LLMPredictor3
from qwen import QWenLLMPredictor
from baichuan import BaichuanLLMPredictor
# from llama2_7b import LLMPredictor6
import yaml


# 将config.yaml转换为Python对象
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


app = FastAPI()
LLM = None

# @app.post("/question_refine")
# async def refine_question(request: Request):
#     json_post_raw = await request.json()
#     json_post = json.dumps(json_post_raw)
#     json_post_list = json.loads(json_post)
#     prompt = json_post_list.get('prompt')
#     history = json_post_list.get('history')
#     # max_length = json_post_list.get('max_length')
#     # top_p = json_post_list.get('top_p')
#     # temperature = json_post_list.get('temperature')
 
#     answer = {
#         "response": "",
#         "history": "",
#         "status": 200,
#         "time": "",
#         "final":False,
#     }
    
#     response = LLM.refine_question(prompt,history)
    
#     answer['response'] = response
#     now = datetime.datetime.now()
#     time = now.strftime("%Y-%m-%d %H:%M:%S")
#     answer["time"]=time

#     log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
#     print(log)
#     return answer


@app.post("/chat")
async def LLM_chat(request: Request):    
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    # max_length = json_post_list.get('max_length')
    # top_p = json_post_list.get('top_p')
    # temperature = json_post_list.get('temperature')
    
    response, history = LLM.reading_comprehension(prompt)
    
    answer = {
        "response": "",
        "history": "",
        "status": 200,
        "time": "",
        "final":False,
    }
    
    answer['response'] = response
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer["time"]=time
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    return answer


@app.post("/batch_chat")
async def LLM_batch_chat(request: Request):    
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    # max_length = json_post_list.get('max_length')
    # top_p = json_post_list.get('top_p')
    # temperature = json_post_list.get('temperature')
    
    response = LLM.batch_reading_comprehension(prompt)
    print(response)
    
    answer = {
        "response": "",
        "history": "",
        "status": 200,
        "time": "",
        "final":False,
    }
    
    answer['response'] =  response
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer["time"]=time
    log = "[" + time + "] " + '", prompt:"' + repr(prompt) + '", response:"' + repr(response) + '"'
    print(log)
    return answer


async def LLM_batch_streamer(json_post_raw):
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    # max_length = json_post_list.get('max_length')
    # top_p = json_post_list.get('top_p')
    # temperature = json_post_list.get('temperature')
    
    answer = {
        "response": "",
        "history": "",
        "status": 200,
        "time": "",
        "final":False,
    }
        
    for response in LLM.batch_reading_comprehension_streamer(prompt):
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer["response"]=response
        answer["history"]=history
        answer["time"]=time
            
        rlt = json.dumps(answer, ensure_ascii = False)
        yield rlt + '\n'
        
    answer['final']=True
    rlt = json.dumps(answer, ensure_ascii = False)
    yield rlt + '\n'
    
async def LLM_streamer(json_post_raw):
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    # max_length = json_post_list.get('max_length')
    # top_p = json_post_list.get('top_p')
    # temperature = json_post_list.get('temperature')
    
    answer = {
        "response": "",
        "history": "",
        "status": 200,
        "time": "",
        "final":False,
    }

        
    for response in LLM.reading_comprehension_streamer(prompt):
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer["response"]=response
        answer["history"]=history
        answer["time"]=time
            
        rlt = json.dumps(answer, ensure_ascii = False)
        yield rlt + '\n'
        
    answer['final']=True
    rlt = json.dumps(answer, ensure_ascii = False)
    yield rlt + '\n'

    
@app.post("/batch_streamer_chat")
async def create_item(request: Request):
    json_post_raw = await request.json()
    return StreamingResponse(LLM_batch_streamer(json_post_raw))


@app.post("/stream_chat")
async def create_batch_item(request: Request):
    json_post_raw = await request.json()
    return StreamingResponse(LLM_streamer(json_post_raw))


@app.post("/")
async def create_item_ori(request: Request):
    json_post_raw = await request.json()
    return StreamingResponse(LLM_streamer(json_post_raw))


def main(args):
    """
    Main function.
    """
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file does not exist! {args.config}")
    
    # 加载config.yaml文件
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
        print(json.dumps(config_dict, ensure_ascii=False, indent=2))
        cfg = Config(**config_dict)

    cuda_device = [i for i in range(torch.cuda.device_count())]
    print(cuda_device)
    # LLM is required as the global variable
    global LLM

    # if cfg.name == 'chatglm':
    #     LLM = LLMPredictor(cuda_device)
    # elif cfg.name == 'chatglm2':
    #     LLM = LLMPredictor2(cuda_device)
    # elif cfg.name == 'bloomz':
    #     LLM = LLMPredictor3(cuda_device)
    if cfg.name == 'qwen':
        LLM = QWenLLMPredictor(cfg=cfg, device_ids=cuda_device)
    elif cfg.name == 'baichuan':
        LLM = BaichuanLLMPredictor(cfg=cfg, device_ids=cuda_device)
    # elif cfg.name == 'llama2':
    #     LLM = LLMPredictor6(cuda_device)

    #LLM.loadmodel(cfg)
    uvicorn.run(app, host='0.0.0.0', port=args.port, workers=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="api")
    parser.add_argument("--port", type=int, default=8080, required=False)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    print(args)
    main(args)


    
