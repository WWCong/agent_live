"""Probe DeepSeek TTS WebSocket: does it synthesize incrementally before finish_text?"""
import asyncio, json, os, time

import websockets

KEY = os.environ["DEEPSEEK_API_KEY"]
URL = "wss://api.deepseek.com/v1/tools/tts?voice_id=stella&format=pcm"
H = {"Authorization": f"Bearer {KEY}"}


def ms(t): return round(t * 1000)


async def session(name, script):
    """script: list of (delay_before_send_s, payload_dict). Logs audio arrival times relative to first send."""
    t0 = time.perf_counter()
    async with websockets.connect(URL, additional_headers=H, open_timeout=15) as ws:
        t_conn = ms(time.perf_counter() - t0)
        start = time.perf_counter()
        sends = []

        async def sender():
            for delay, payload in script:
                if delay:
                    await asyncio.sleep(delay)
                await ws.send(json.dumps(payload, ensure_ascii=False))
                sends.append((ms(time.perf_counter() - start), payload.get("event"), payload.get("text", "")))

        task = asyncio.create_task(sender())
        frames = []  # (t_ms, bytes)
        final = None
        async for m in ws:
            if isinstance(m, bytes):
                frames.append((ms(time.perf_counter() - start), len(m)))
            else:
                final = json.loads(m)
                break
        await task
    first = frames[0][0] if frames else None
    finish_sent = next((t for t, e, _ in sends if e == "finish_text"), None)
    audio_s = round(sum(n for _, n in frames) / 48000, 2)
    # audio received (seconds) by the time finish_text was sent
    before_finish = round(sum(n for t, n in frames if finish_sent is not None and t < finish_sent) / 48000, 2)
    print(f"[{name}] connect={t_conn}ms first_audio=+{first}ms finish_text_sent=+{finish_sent}ms "
          f"audio_before_finish={before_finish}s total_audio={audio_s}s frames={len(frames)} "
          f"last_frame=+{frames[-1][0] if frames else None}ms final={final}")
    return first


async def main():
    s1 = "你好，我是你的语音助手。"
    s2 = "今天想聊点什么呢？"
    s3 = "我可以帮你查天气、订闹钟，或者陪你聊聊天。"

    # A: whole text + finish immediately (baseline first-packet over WS)
    for i in range(3):
        await session(f"A{i} all+finish", [(0, {"event": "text", "text": s1 + s2}), (0, {"event": "finish_text"})])

    # B: one full sentence, then wait 4 s before finishing -> is audio produced before finish_text?
    await session("B sentence, wait 4s", [(0, {"event": "text", "text": s1}),
                                          (4.0, {"event": "text", "text": s2}),
                                          (0, {"event": "finish_text"})])

    # C: LLM-like trickle: 2 chars every 40 ms over three sentences, then finish
    text = s1 + s2 + s3
    script = [(0 if i == 0 else 0.04, {"event": "text", "text": text[i:i + 2]}) for i in range(0, len(text), 2)]
    script.append((0, {"event": "finish_text"}))
    await session("C trickle 2ch/40ms", script)

    # D: three sentences, 1.5 s apart (simulates slow generation) then finish
    await session("D 3 sentences 1.5s apart", [(0, {"event": "text", "text": s1}),
                                               (1.5, {"event": "text", "text": s2}),
                                               (1.5, {"event": "text", "text": s3}),
                                               (0, {"event": "finish_text"})])


if __name__ == "__main__":
    asyncio.run(main())
