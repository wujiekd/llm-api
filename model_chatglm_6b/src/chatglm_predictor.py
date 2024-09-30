import json
import sys
import os
import os.path
import torch
import random
import numpy as np
import logging as logger
from model_chatglm_6b.src.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig
from model_chatglm_6b.src.tokenization_chatglm import ChatGLMTokenizer
from model_chatglm_6b.src.lora import convert_to_lora_recursively, print_trainable_parameters, mark_only_lora_as_trainable
import time

class ChatGLMPredictor():
    def __init__(self, model_path):
        self.set_random_seed(42)
        self.config = ChatGLMConfig.from_json_file(os.path.join(model_path, "config.json"))
        self.tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
        self.model = ChatGLMForConditionalGeneration(self.config)
        convert_to_lora_recursively(self.model, lora_rank=8, lora_alpha=8, lora_dropout=0.1)
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu'), strict=True)
        self.model = self.model.half().cuda()
        self.seq_len = 2048
        self.idx = 0

    def to(self, device):
        return self.model.to(device)

    # Fix seed and predict deterministically
    def set_random_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    # Save result
    def save_result(self, record, dic, save):
        if save:
            save_dir = 'record' + str(record)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            f = open(save_dir + '/' + str(self.idx).zfill(8) + '.json', 'w')
            f.write(json.dumps(dic, ensure_ascii=False) + '\n')
            f.close()
        self.idx += 1

    # Forward
    def predict(self, dic, seq_len=2048, record=0, save=False):
        self.set_random_seed(42)
        # question
        question = dic['llm_input']
        # tokens_list = self.tokenizer.encode(question, add_special_tokens=False)

        # # add special tokens
        # tokens_list = self.tokenizer.build_inputs_with_special_tokens(tokens_list)

        # input_dict = {
        #     "input_ids":torch.tensor(tokens_list, dtype=torch.long).unsqueeze(0).cuda(),
        # }
        start = time.time()
        answer, history = self.model.stream_chat(self.tokenizer, question, history=[], max_new_tokens=seq_len, do_sample=False)
        print("[InputLength]:", len(question))
        print('[Input]:\n', question)
        print('[Output]:\n', answer)
        print('[Time]:', time.time() - start)
        print('\n')
        dic['time'] = time.time() - start
        # dic['question'] = question
        dic['answer'] = answer
        self.save_result(record, dic, save)
        return dic
