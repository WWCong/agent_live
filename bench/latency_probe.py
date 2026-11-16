"""Small latency probe for DeepSeek ASR / TTS / deepseek-flash. Prints timings only (never the key)."""
import asyncio, io, json, os, statistics, sys, time, wave

import httpx

KEY = os.environ["DEEPSEEK_API_KEY"]
API = "https://api.deepseek.com"
H = {"Authorization": f"Bearer {KEY}"}
OUT = os.path.dirname(os.path.abspath(__file__))
N = int(os.environ.get("N", "5"))


def ms(t): return round(t * 1000)


def summary(name, xs):
    xs = sorted(xs)
    print(f"  {name}: n={len(xs)} min={xs[0]} p50={statistics.median(xs)} max={xs[-1]} all={xs}")


async def rtt(c):
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        r = await c.get(f"{API}/models", headers=H)
        ts.append(ms(time.perf_counter() - t0))
    print(f"[baseline] GET /models status={r.status_code}")
    summary("round-trip ms (warm connection)", ts)


async def llm(c, thinking, runs):
    ttft, total, out_chars = [], [], []
    msgs = [
        {"role": "system", "content": "你是语音助手。用口语回答，最多两句话，不要用 Markdown。"},
        {"role": "user", "content": "周末想在家附近放松一下，有什么建议？"},
    ]
    for _ in range(runs):
        body = {"model": "deepseek-flash", "messages": msgs, "stream": True, "max_tokens": 200,
                "thinking": {"type": thinking}}
        t0 = time.perf_counter(); first = None; text = ""; first_reasoning = None
        async with c.stream("POST", f"{API}/chat/completions", json=body, headers=H) as r:
            if r.status_code != 200:
                print("  LLM error", r.status_code, (await r.aread())[:300]); return
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                if not d.get("choices"):
                    continue
                delta = d["choices"][0]["delta"]
                if delta.get("reasoning_content") and first_reasoning is None:
                    first_reasoning = time.perf_counter() - t0
                if delta.get("content"):
                    if first is None:
                        first = time.perf_counter() - t0
                    text += delta["content"]
        ttft.append(ms(first)); total.append(ms(time.perf_counter() - t0)); out_chars.append(len(text))
    print(f"[LLM deepseek-flash thinking={thinking}] sample reply: {text[:80]!r}")
    summary("time to first answer token ms", ttft)
    summary("total ms", total)
    summary("answer chars", out_chars)


async def tts_http(c, text, runs, save=None, fmt="pcm"):
    ttfb, total, dur = [], [], []
    for i in range(runs):
        t0 = time.perf_counter(); first = None; buf = b""
        async with c.stream("POST", f"{API}/v1/tools/tts", headers=H,
                            json={"text": text, "voice_id": "stella", "format": fmt}) as r:
            if r.status_code != 200:
                print("  TTS error", r.status_code, (await r.aread())[:300]); return None
            ctype = r.headers.get("content-type")
            async for chunk in r.aiter_bytes():
                if first is None and chunk:
                    first = time.perf_counter() - t0
                buf += chunk
        ttfb.append(ms(first)); total.append(ms(time.perf_counter() - t0))
        audio_bytes = len(buf) - (44 if fmt == "wav" else 0)
        dur.append(round(audio_bytes / 2 / 24000, 2))
    print(f"[TTS HTTP format={fmt}] text={len(text)} chars content-type={ctype}")
    summary("first audio byte ms", ttfb)
    summary("total ms", total)
    summary("audio seconds", dur)
    if save:
        with open(os.path.join(OUT, save), "wb") as f:
            f.write(buf)
    return buf


def pcm24_to_wav(pcm):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm)
    return b.getvalue()


async def asr(c, wav_bytes, label, runs):
    lat = []; res = None
    for _ in range(runs):
        t0 = time.perf_counter()
        r = await c.post(f"{API}/v1/tools/asr", headers=H, files={"file": ("a.wav", wav_bytes, "audio/wav")})
        lat.append(ms(time.perf_counter() - t0))
        if r.status_code != 200:
            print("  ASR error", r.status_code, r.text[:300]); return
        res = r.json()
    print(f"[ASR {label}] {len(wav_bytes)//1024} KiB, response keys={sorted(res)} duration_ms={res['duration_ms']} text={res['text']!r}")
    summary("latency ms", lat)


async def tts_ws():
    try:
        import websockets
    except ImportError:
        print("[TTS WS] websockets not installed"); return
    url = "wss://api.deepseek.com/v1/tools/tts?voice_id=stella&format=pcm"
    t0 = time.perf_counter()
    try:
        async with websockets.connect(url, additional_headers=H, open_timeout=15) as ws:
            t_conn = time.perf_counter() - t0
            t1 = time.perf_counter()
            await ws.send(json.dumps({"event": "text", "text": "你好，我是你的语音助手。"}, ensure_ascii=False))
            first = None; nbytes = 0; finished = None; sent_rest = False
            async def rest():
                await asyncio.sleep(2.0)
                await ws.send(json.dumps({"event": "text", "text": "今天想聊点什么呢？"}, ensure_ascii=False))
                await ws.send(json.dumps({"event": "finish_text"}))
            task = asyncio.create_task(rest())
            async for m in ws:
                if isinstance(m, bytes):
                    if first is None:
                        first = time.perf_counter() - t1
                    nbytes += len(m)
                else:
                    finished = json.loads(m); break
            await task
        print(f"[TTS WS] connect ms={ms(t_conn)} first audio after first text ms={ms(first) if first else None} "
              f"(finish_text sent at +2000ms) audio_s={round(nbytes/48000,2)} final={finished}")
    except Exception as e:
        print(f"[TTS WS] failed after {ms(time.perf_counter()-t0)} ms: {type(e).__name__}: {e}")


async def main():
    async with httpx.AsyncClient(timeout=60, http2=False) as c:
        await rtt(c)
        await llm(c, "disabled", N)
        await llm(c, "enabled", 2)
        short = "好的，我帮你查一下明天上海的天气。"
        await tts_http(c, short, N)
        pcm = await tts_http(c, "周末可以去附近的公园散散步，或者找家安静的咖啡馆看看书，都挺放松的。", 1, save="sample.pcm")
        if pcm:
            wav = pcm24_to_wav(pcm)
            await asr(c, wav, f"{round(len(pcm)/48000,1)}s 24kHz wav", N)
            short_pcm = await tts_http(c, "明天上海天气怎么样？", 1)
            if short_pcm:
                await asr(c, pcm24_to_wav(short_pcm), f"{round(len(short_pcm)/48000,1)}s 24kHz wav", N)
    await tts_ws()


if __name__ == "__main__":
    asyncio.run(main())
