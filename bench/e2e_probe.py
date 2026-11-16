"""端到端探针：用户文本 → deepseek-flash 流式 → 分句并发 TTS → 模拟实时播放，统计首段音频时间和播放断档。"""
import asyncio, time, statistics, httpx
import pipeline as s
SYS = {"role": "system", "content": "你是语音助手。用口语回答，两三句话，不要用 Markdown。"}
QS = ["周末想在家附近放松一下，有什么建议？", "怎么才能睡得更好？", "给我讲个简短的冷知识。"]
async def one(http, q):
    hist = [SYS]; t0 = time.perf_counter(); st = {"first": None, "playhead": 0.0, "gaps": 0.0, "bytes": 0}
    async def play(pcm):
        now = time.perf_counter() - t0
        if st["first"] is None: st["first"] = now; st["playhead"] = now
        if now > st["playhead"] + 0.02: st["gaps"] += now - st["playhead"]; st["playhead"] = now
        st["playhead"] += len(pcm) / 48000; st["bytes"] += len(pcm)
    # also record chunk boundaries
    chunks = []
    orig = s.chunk_for_tts
    async def spy(deltas):
        async for c in orig(deltas):
            chunks.append((round((time.perf_counter()-t0)*1000), c)); yield c
    s_chunk = s.chunk_for_tts; s.chunk_for_tts = spy
    try: await s.respond(http, hist, q, play)
    finally: s.chunk_for_tts = s_chunk
    print(f"Q={q}\n  first_chunk_ready={chunks[0][0]}ms first_audio={round(st['first']*1000)}ms "
          f"audio={round(st['bytes']/48000,1)}s playback_gaps={round(st['gaps']*1000)}ms chunks={[(t,c) for t,c in chunks]}")
    return st["first"]
async def main():
    async with httpx.AsyncClient(timeout=60) as http:
        await http.get(f"{s.API}/models", headers=s.AUTH)  # warm the connection
        firsts = [await one(http, q) for q in QS]
        print("first audio p50 ms:", round(statistics.median(firsts)*1000))
        # ASR helper smoke test on 16 kHz audio produced by TTS (decimate 24k->16k crudely)
        pcm24 = b"".join([c async for c in s.tts_pcm(http, "今天下午三点提醒我开会。")])
        import array; a = array.array("h"); a.frombytes(pcm24)
        pcm16 = array.array("h", (a[int(i*1.5)] for i in range(len(a)*2//3))).tobytes()
        t0 = time.perf_counter(); txt = await s.asr(http, pcm16)
        print(f"asr() -> {txt!r} in {round((time.perf_counter()-t0)*1000)}ms")
if __name__ == "__main__":
    asyncio.run(main())
