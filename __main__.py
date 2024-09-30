import argparse
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn, json, datetime
from chatglm import LLMPredictor
from chatglm2 import LLMPredictor2
from bloomz import LLMPredictor3
from qwen import LLMPredictor4
from baichuan import LLMPredictor5
from llama2_7b import LLMPredictor6
import yaml
