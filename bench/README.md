# DeepSeek 语音接口延迟探针

需要 Python 3.10+ 和 `DEEPSEEK_API_KEY` 环境变量。

```bash
pip install -r bench/requirements.txt
```

```bash
cd bench && python3 latency_probe.py
```

```bash
cd bench && python3 tts_ws_probe.py
```

```bash
cd bench && python3 e2e_probe.py
```

| 脚本 | 测什么 |
|---|---|
| `latency_probe.py` | 网络往返基线；deepseek-flash 首 token（关闭和开启思考）；TTS HTTP 首包；ASR 2 秒和 7 秒语音的耗时 |
| `tts_ws_probe.py` | TTS WebSocket 在 `finish_text` 之前会不会出音频（即是否增量合成） |
| `e2e_probe.py` | 用户文本 → LLM → 分句 → 逐句并发 TTS → 模拟实时播放，统计首段音频时间和播放断档 |
| `pipeline.py` | 上面用到的管线核心函数，可以直接复用 |

跑一遍全部脚本大约花 0.05 元。结果受网络影响很大，**请在实际部署的机房里跑**，并把 `latency_probe.py` 打印的往返基线一并记下来。
