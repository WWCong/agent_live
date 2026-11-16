"""DeepSeek 级联语音管线的核心函数（ASR / LLM 流式 / 分句 / 逐句 TTS / 按序播放）。2026-11-16 已用真实接口跑通。"""
import asyncio, io, json, os, wave
import httpx

KEY = os.environ["DEEPSEEK_API_KEY"]
API = "https://api.deepseek.com"
AUTH = {"Authorization": f"Bearer {KEY}"}

def pcm_to_wav(pcm: bytes, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    return buf.getvalue()

async def asr(http: httpx.AsyncClient, pcm16k: bytes) -> str:
    """一段话（≤120 s）整段上传，返回完整文本。"""
    r = await http.post(f"{API}/v1/tools/asr", headers=AUTH,
                        files={"file": ("utt.wav", pcm_to_wav(pcm16k), "audio/wav")})
    r.raise_for_status()
    return r.json()["text"].strip()

async def llm_stream(http: httpx.AsyncClient, messages):
    """OpenAI 兼容流式接口，关闭思考，逐个 yield 文本增量。"""
    body = {"model": "deepseek-flash", "messages": messages, "stream": True,
            "max_tokens": 300, "thinking": {"type": "disabled"}}
    async with http.stream("POST", f"{API}/chat/completions", json=body, headers=AUTH) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choices = json.loads(line[6:]).get("choices") or [{}]
            delta = choices[0].get("delta", {}).get("content")
            if delta:
                yield delta

SOFT, HARD = "，,、：:", "。！？!?；;\n"

async def chunk_for_tts(deltas, first_min=6):
    """第一段到逗号且不少于 first_min 字就送出；之后按整句切。"""
    buf, first = "", True
    async for d in deltas:
        buf += d.replace("*", "").replace("#", "").replace("`", "")
        marks = SOFT + HARD if first else HARD
        cut = max(buf.rfind(p) for p in marks)
        if cut >= 0 and (not first or cut + 1 >= first_min):
            yield buf[:cut + 1]; buf = buf[cut + 1:]; first = False
    if buf.strip():
        yield buf

async def tts_pcm(http: httpx.AsyncClient, text: str):
    """一个分句一次 HTTP 请求，流式返回 24 kHz s16le 单声道 PCM。"""
    async with http.stream("POST", f"{API}/v1/tools/tts", headers=AUTH,
                           json={"text": text, "voice_id": "stella", "format": "pcm"}) as r:
        r.raise_for_status()
        async for chunk in r.aiter_bytes():
            yield chunk

async def speak(http: httpx.AsyncClient, chunks, play):
    """分句一出来就并发发起 TTS 请求，音频严格按顺序播放。"""
    order: asyncio.Queue = asyncio.Queue()
    tasks = []

    async def fetch(text, q):
        try:
            async for pcm in tts_pcm(http, text):
                await q.put(pcm)
        finally:
            await q.put(None)

    async def producer():
        try:
            async for text in chunks:
                q = asyncio.Queue()
                tasks.append(asyncio.create_task(fetch(text, q)))
                await order.put(q)
        finally:
            await order.put(None)

    prod = asyncio.create_task(producer())
    try:
        while (q := await order.get()) is not None:
            while (pcm := await q.get()) is not None:
                await play(pcm)
    finally:                      # 被打断时外层 cancel 本协程，这里统一清理
        prod.cancel()
        for t in tasks:
            t.cancel()

async def respond(http, history, user_text, play):
    history.append({"role": "user", "content": user_text})
    reply = []
    async def tee():
        async for d in llm_stream(http, history):
            reply.append(d); yield d
    try:
        await speak(http, chunk_for_tts(tee()), play)
    finally:                      # 被打断时这里只记下已生成的部分；按已播放时长截断见 3.3 节 ⑥
        history.append({"role": "assistant", "content": "".join(reply)})
