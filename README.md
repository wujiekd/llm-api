## 大语言模型部署API

定制化修改API，支持单条输入和批量输入，支持非流式和流式

支持chatglm, chatglm2, bloomz, qwen, baichuan, llama2

## 启动api
```sh
sh run_api.sh
```

## 推理，批量推理
```sh
sh run_predict.sh
```

## 指定显卡启动api
```python
CUDA_VISIBLE_DEVICES=0,1 sh run_api.sh
```

#### curl测试命令 部分模型支持batch推理
```sh
curl -X POST "http://0.0.0.0:8081/chat" -H 'Content-Type: application/json' -d '{"prompt": "你晚上吃饭了吗？"}'

curl -X POST "http://0.0.0.0:8081/stream_chat" -H 'Content-Type: application/json' -d '{"prompt": "你晚上吃饭了吗？"}'

curl -X POST "http://0.0.0.0:8081/batch_chat" -H 'Content-Type: application/json' -d '{"prompt": ["你晚上吃饭了吗？","你中午吃饭了吗？","如何处理隐私问题"]}'

curl -X POST "http://0.0.0.0:8081/batch_streamer_chat" -H 'Content-Type: application/json' -d '{"prompt": ["你晚上吃饭了吗？","你中午吃饭了吗？","如何处理隐私问题"]}'

```

