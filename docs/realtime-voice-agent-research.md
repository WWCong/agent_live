# 实时语音对话 Agent：开源方案调研 + 基于 DeepSeek ASR/TTS 的设计

> 调研和实测日期：2026-11-16。DeepSeek 语音接口 2026-11-13 发布，之后可能还会调整。
>
> **可信度标注**
> - ✅ 已核实：读过官方文档或官方仓库源码。DeepSeek 文档读的是 `api-docs.deepseek.com` 的 Markdown 版本。
> - 📏 实测：用真实 API Key 测量。
> - 🟡 二手：来自第三方或媒体报道。
>
> **实测环境说明**：测试机是海外云容器，经过代理访问 `api.deepseek.com`，网络往返约 **0.36–0.40 s**。
> 每个 HTTP 请求至少要付出 1 次往返，所以下文的绝对数字偏大。国内机房部署时，大致每个请求可以减去约 0.35 s。
> 测量脚本在 [`bench/`](../bench/README.md)，建议在实际部署的机房复测。

---

## 0. 结论速览

1. **你说的阿里开源语音 Agent 是 [`qwen-audio-agent`](https://github.com/QwenAudio/qwen-audio-agent)** ✅
   - 组织是 QwenAudio，Apache-2.0，2026-07-28 开源，最新版本 v2.1.0（2026-11-10）。
   - 它不是模型，而是"全双工语音前台 + 后台 Agent"的运行时。前台由云端实时语音模型负责，默认 `qwen-audio-3.1-realtime-plus`。后台通过 ACP 协议把任务委派给编码 Agent。
   - 前台不支持 DeepSeek；DeepSeek 只能作为后台（DeepSeek Harness）。
2. **DeepSeek 的语音接口只能用来搭"级联式"管线** ✅📏
   - **ASR 不支持流式**：整段上传，单段最长 120 s、最大 8 MiB。
   - **TTS 有两种模式**：HTTP 模式边合成边返回音频；WebSocket 模式可以分块发送文本。
   - **但实测 WebSocket 在收到 `finish_text` 之前不返回任何音频**。即使先送 108 个字、多个完整句子再等 5 秒，也一样。所以目前它**不能做到"LLM 边生成、TTS 边合成"**。
   - TTS 首包时间与文本长度无关：3 个字和 108 个字的首包都是约 1.2–1.4 s。
3. **实测延迟（本测试机）** 📏
   - LLM 首 token：中位数 0.93 s（关闭思考）；开启思考约 1.58 s。
   - TTS 首包：约 1.3–1.4 s。
   - ASR：2 秒语音约 1.0–1.2 s，7 秒语音约 1.7 s。
   - **从用户文本到听到第一段音频**，端到端中位数 **2.3 s**。
   - 再加上判停等待和 ASR，**从用户说完到听到回答约 4 s**。按国内机房估算约 **2.5–3 s**。
   - 这明显慢于端到端实时语音模型（0.3–0.8 s）。
4. **推荐方案**：Pipecat 1.16 + 三个 DeepSeek 服务。
   - LLM 直接用 Pipecat 内置的 `DeepSeekLLMService`。
   - ASR 写一个 `SegmentedSTTService` 子类。
   - TTS 写一个**基于 HTTP 的 `TTSService` 子类，开启按句聚合**。
   - 体验上要靠三个手段：
     1. 预合成语气词音频，判停后立即播放，掩盖等待；
     2. 投机 ASR 和投机 LLM；
     3. 第一句尽快送 TTS。
   - 延迟要求严格时，把 TTS 换成低首包的流式 TTS，或者换用端到端模型做前台（见 3.6）。
5. **成本** ✅：TTS 是大头。限时五折下，机器人每说 1 分钟约 0.08 元；ASR 每分钟约 0.0065 元；`deepseek-flash` 每轮约 0.001–0.003 元。**10 分钟通话约 0.5 元**。

---

## 1. 开源方案全景

### 1.1 两条技术路线

| | 级联式（ASR → LLM → TTS） | 端到端 / 全双工语音模型 |
|---|---|---|
| 代表 | Pipecat、LiveKit Agents、TEN，加任意 ASR/LLM/TTS | Qwen-Omni 系列、Fun-Audio-Chat、Kimi-Audio、Step-Audio 2、MiniCPM-o 4.5、Moshi、GLM-4-Voice |
| 首字延迟 | 约 0.7–3 s，取决于各段供应商 | 约 0.3–0.8 s |
| LLM 智力和工具调用 | 可以接最强的文本模型，工具生态成熟 | 受语音模型能力限制，工具调用能力参差 |
| 副语言（情绪、语气、笑声） | 在 ASR 这一步丢失 | 能保留 |
| 可控性和调试 | 每段都可以替换、记日志 | 更接近黑盒 |
| **DeepSeek 适用性** | **唯一可行的方式** | DeepSeek 只能做后台的"慢思考" |

业界常见做法是两者组合：端到端实时模型做前台快速对话，需要深度推理或工具的任务交给后台文本 Agent。qwen-audio-agent 走的就是这条路。

### 1.2 阿里系

| 项目 | 类型 | 开源情况 | 要点 |
|---|---|---|---|
| **[qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent)** ✅ | 语音 Agent 运行时（Node.js） | Apache-2.0 | 见下方详述 |
| [Qwen-Live-Harness](https://github.com/QwenLM/Qwen-Live-Harness) ✅ | 桌面 Agent（Electron + Node 守护进程） | Apache-2.0 | 用 Qwen Omni Realtime API "看、听、做"，内置记忆；后台通过 ACP 委派给 Qwen Code、Qoder、Codex、Claude Code、Gemini CLI。目前只支持 macOS，需要 DashScope Key |
| Qwen-Audio-3.0/3.1-Realtime 🟡 | 实时语音大模型 | **仅提供 API**，没有找到开放权重的说明 | 2026-07 发布 3.0，分 Plus 和 Flash 两档；能自主调用工具；内置"多模态感知双工控制"子模型判断对话节奏 |
| Qwen3-Omni / Qwen-Omni 系列 🟡 | 端到端全模态模型 | 开源权重 | 可以本地部署语音对话 |
| Fun-Audio-Chat 🟡 | 端到端语音对话模型 | 开源（论文 arXiv 2512.20156） | 5 Hz 低帧率共享 LLM 主干，加 25 Hz 语音头，降低算力开销 |
| Qwen3-ASR 🟡 | ASR | Apache-2.0，0.6B 和 1.7B | 多语种、多方言；**流式推理目前只在 vLLM 后端可用** |
| FunASR / Fun-ASR-Nano-2512 🟡 | ASR 工具包 | 开源 | 有 paraformer 流式模型和 2-pass 方案；2026-05 加入 vLLM 加速和带 VAD 的流式 WebSocket 服务。Pipecat 已内置 `FunASRSTTService` |
| SenseVoice 🟡 | ASR 加情感和声音事件识别 | 开源 | 非自回归，速度很快 |
| CosyVoice 2/3 🟡 | TTS | 开源 | 支持流式，第三方称流式首包约 150 ms |
| Qwen3-TTS 🟡 | TTS | 开源，0.6B 和 1.7B | 宣称输入一个字就能输出首个音频包 |

#### qwen-audio-agent 详述 ✅

- **三部分架构**，由 Gateway 托管，提供 Web UI、终端 TUI 和桌面悬浮球：
  - 前台 Agent：实时语音模型，加上指令、上下文和轻量工具；
  - 编排运行时：管理任务、权限、会话和事件；
  - 后台 Agent：一个 ACP 子进程，在用户环境里干活。
- **非阻塞委派**：
  - 前台调用 `spawn_thinking(objective)` 后**立即返回**，用户可以继续说话；
  - 任务进入有界队列，默认全局只有 1 个活跃槽位；
  - 结果要等到"安全的双工插话窗口"才播报；
  - `objective` 是前台模型**改写过的自包含指令**，不是 ASR 原文。
- **前台工具**：
  - `update_agent_task`、`get_agent_task_status`、`cancel_agent_task`、`respond_permission`，以及记忆、搜索等可选工具。
  - 查状态和取消都直接读任务记录，不经过 LLM；取消要等后台确认，不能只凭模型口头说"已取消"。
- **打断和权限**：
  - 用户打断只推迟播报，**不取消已提交的后台任务**；
  - 权限审批必须来自用户**当前这一轮**的明确表态。
- **可插拔的实时 Provider**：
  - 云端：Qwen Audio 3.0/3.1 Realtime、GPT-Live/OpenAI Realtime、Gemini Live、Qwen3.5-Omni、Qwen3.8 Omni Flash、豆包 Seeduplex 3.0、StepAudio 3 Realtime；
  - 本地或云端：MiniCPM-o 4.5；
  - 本地级联：**Hugging Face speech-to-speech**（VAD + STT + LLM + TTS），对外暴露兼容 OpenAI Realtime 的 WebSocket，地址是 `ws://127.0.0.1:8765/v1/realtime`。
- **后台 Agent**：
  - 原生 ACP：Qwen Code、OpenCode、Qoder、Kimi Code、Hermes、CodeBuddy、**DeepSeek Harness**、OpenClaw；
  - 通过适配器：Codex、Claude Code。
- **安装**：

  ```bash
  npm i -g qwen-audio-agent
  ```

  然后运行 `qwenaudio config`，再启动 `qwenaudio`，最后打开 `qwenaudio webui` 或 `qwenaudio tui`。

### 1.3 编排框架（级联式）

| 框架 | 许可 | 语言 | 适合场景 |
|---|---|---|---|
| **[Pipecat](https://github.com/pipecat-ai/pipecat)** ✅（读过 1.16.0 源码） | BSD-2 | Python | 帧流水线结构，集成的服务最多。**已内置 DeepSeek LLM、FunASR、Silero VAD、Smart Turn v2/v3**，传输层支持 SmallWebRTC、WebSocket、Daily、LiveKit |
| [LiveKit Agents](https://github.com/livekit/agents) | Apache-2.0 | Python 和 Node | 基于 WebRTC SFU，适合多人、电话、视频；自带判停模型 |
| [TEN Framework](https://github.com/TEN-framework/ten-framework) | 需自行核实（各来源说法不一） | 多语言图编排 | 适合多模态和数字人；TEN VAD（MIT）和判停模型可以单独使用 |
| [HF speech-to-speech](https://github.com/huggingface/speech-to-speech) | Apache-2.0 | Python | 纯本地级联，可以作为 qwen-audio-agent 的 Provider |

### 1.4 其他端到端开源模型 🟡

- Kimi-Audio、Step-Audio 2、GLM-4-Voice、Moshi
- MiniCPM-o 4.5：全双工，但不支持工具调用
- NVIDIA PersonaPlex：基于 Moshi
- NVIDIA NemotronLabs VoiceChat 11B：号称首个支持实时工具调用的开源全双工模型，轮换延迟 448 ms

### 1.5 DeepSeek 生态现状

- [deepseek-ai/chat-demo](https://github.com/deepseek-ai/chat-demo) ✅：官方前端，MIT。语音功能只有"录音后识别"和"整段朗读"，**不是实时通话**。
- [hangi7890/deepseek-live](https://github.com/hangi7890/deepseek-live) 🟡：可打断，但用的是浏览器自带的 STT。
- [dsh-voice-talk](https://github.com/duoduoqian708/dsh-voice-talk) 和 [dsh-voice-call](https://github.com/biliye/dsh-voice-call) 🟡：TTS 用的是千问、讯飞或系统语音。
- 目前**没有看到**用 DeepSeek 官方 ASR 和 TTS 做的成熟实时对话开源项目。

### 1.6 选型建议

| 你的目标 | 推荐 |
|---|---|
| 用 DeepSeek 全家桶做语音助手，能接受 2–3 s 延迟 | Pipecat + DeepSeek 三件套（第 3 节） |
| 想要 DeepSeek 的大脑，同时延迟尽量低 | DeepSeek LLM（加 ASR）+ 低首包流式 TTS（CosyVoice 自部署或其他实时 TTS） |
| 给编码 Agent 加"打电话式"语音前台 | qwen-audio-agent，后台选 DeepSeek Harness |
| 追求最低延迟和情感表现 | 前台用端到端实时模型，DeepSeek 做后台 |
| 完全离线 | FunASR 或 Qwen3-ASR + 本地 LLM + CosyVoice 或 Qwen3-TTS，用 Pipecat 编排；或直接用端到端开源模型 |

---

## 2. DeepSeek 语音相关接口（官方文档 ✅，加实测 📏）

### 2.1 ASR：`POST https://api.deepseek.com/v1/tools/asr`

| 项 | 内容 |
|---|---|
| 鉴权 | `Authorization: Bearer <key>` |
| 请求 | `multipart/form-data`，只有一个字段 `file`。**没有** model、语种、热词参数 |
| 格式 | 16-bit PCM WAV 或 Ogg Opus，根据文件内容判断。WAV 采样率支持 8000、16000、22050、24000、32000、44100、48000 Hz；双声道只识别左声道 |
| 限制 | 最大 8 MiB，上传须在 10 分钟内完成，**最长 120 s** |
| 流式 | **不支持**，整段提交，返回完整文本 |
| 返回 | `{"request_id": "...", "text": "...", "duration_ms": 3200}`（📏 实测三个字段都存在） |
| 语种 | 21 种，自动识别：中、粤、英、日、韩、德、法、西、葡、意、俄、阿、印地、印尼、越、泰、土、马来、菲、波兰、匈牙利 |
| 错误码 | 400 格式错误或缺少 `file`；408 上传超时；413 超过 8 MiB；**429 并发超限**；503 服务繁忙；504 识别超时 |
| 限流 | **每个账号最多同时 50 个请求** |
| 价格 | 0.39 元/小时（限时五折，原价 0.78 元），按秒计费，不足 1 秒按 1 秒算 |

### 2.2 TTS（HTTP）：`POST https://api.deepseek.com/v1/tools/tts`

| 字段 | 必填 | 说明 |
|---|---|---|
| `text` | 是 | 不能为空，最多 10,000 计费字符；请求体最大 64 KiB |
| `voice_id` | 否 | `stella`（女声，默认）或 `tide`（男声） |
| `format` | 否 | `wav`（默认）或 `pcm`。`pcm` 是 24 kHz、16-bit 有符号、单声道 |
| `filter_markdown` | 否 | 朗读前去掉 Markdown 格式，默认 `true` |

- **流式返回**：音频边合成边返回。
  - 📏 `pcm` 格式的响应头是 `Content-Type: audio/pcm; rate=24000; encoding=signed-integer; bits=16; channels=1`。
  - 📏 合成速度约为实时的 7 倍：24 s 的音频在首包之后约 3.5 s 内全部到达。
- **中途失败**：如果已经开始返回音频后合成失败，响应会被截断，**提前结束的响应应当按失败处理**。
- **不支持**：调语速、断点续传。
- **语种**：10 种，根据文本自动识别：中、粤、英、日、韩、马来、阿、印尼、泰、越。
- **错误码**：
  - 400：没有可读的文本，或文本未通过内容审核（`Content Exists Risk`）；
  - 429：并发超限；
  - 503：服务繁忙；
  - 504：合成超时。
- **限流**：每个账号最多同时 50 个请求。
- **价格**：1.5 元 / 万字符（限时五折，原价 3 元）。**一个汉字算 2 个字符**，其他字符各算 1 个。

### 2.3 TTS（WebSocket）：`wss://api.deepseek.com/v1/tools/tts`

- **查询参数**：
  - `voice_id`；
  - `format`：默认 `pcm`，也可以选 `opus`。`opus` 返回的是不带容器的 Opus 包，不能直接存成 Ogg；
  - `filter_markdown`。
- **鉴权**（二选一）：
  - 请求头 `Authorization: Bearer <key>`；
  - 子协议 `Sec-WebSocket-Protocol: tts, bearer.<key>`。
- **客户端发送**：每段文本发 `{"event": "text", "text": "..."}`，最后一段之后发 `{"event": "finish_text"}`。
- **服务端返回**：
  - 二进制帧是音频。📏 每帧约 0.1 s（4800 字节）。
  - 结束时发一条文本帧 `{"event": "finish", "code": "...", "message": "..."}`。`code` 为 `success` 表示成功，失败时可能是 `invalid_input`、`content_filter`、`input_timeout`、`service_error`。📏 实测成功时收到的是 `{"event":"finish","code":"success","message":"completed"}`。
- **限制**：
  - 60 s 内没有收到文本，会话结束；
  - 单个会话最长 30 分钟、最多 10,000 计费字符、64 KiB；
  - 服务端每 25 s 发一次 ping。

#### 📏 关键实测：WebSocket 在 `finish_text` 之前不出音频

| 场景 | `finish_text` 发出时刻 | 第一帧音频 | `finish_text` 之前收到的音频 |
|---|---|---|---|
| A. 整段文本加 `finish_text` 立即发出（3 次） | +0 ms | +1.05 / 1.19 / 1.79 s | — |
| B. 一个完整句子，等 4 s 再发后续文本和 `finish_text` | +4.0 s | +5.27 s | **0 s** |
| C. 模拟 LLM 输出，每 40 ms 发 2 个字，共 3 句 | +0.81 s | +1.99 / 2.31 s | **0 s** |
| D. 3 个句子，每隔 1.5 s 发一句 | +3.0 s | +4.05 / 4.32 s | **0 s** |
| E. 108 字多句长文本，等 5 s 再发 `finish_text` | +5.0 s | +6.19 s | **0 s** |

- **结论**：至少在 2026-11-16 实测时，服务端是**等收齐全部文本才开始合成**，没有"按句边收边合成"。
- WebSocket 建连本身还要 0.9–1.5 s（本测试机经代理）。
- 因此**实时对话不应该把整轮回复塞进一个 WebSocket 会话**。应该**每个分句单独发一个 HTTP 请求**，利用 keep-alive 复用连接，并发获取，按顺序播放。
- 等官方支持真正的增量合成后，再切回 WebSocket。复测脚本是 `bench/tts_ws_probe.py`。

### 2.4 LLM：`deepseek-flash`

- **模型**：`deepseek-flash`，即 DeepSeek-V4.1-Flash-1017。上下文 1M，最大输出 384K，并发上限 2500。
- **三种协议都可用** ✅：
  - Chat Completions：`POST https://api.deepseek.com/chat/completions`
  - Anthropic Messages：`POST https://api.deepseek.com/anthropic/v1/messages`
  - Responses：`POST https://api.deepseek.com/responses`
- **思考模式默认开启**：
  - OpenAI 和 Anthropic 格式用 `{"thinking": {"type": "disabled"}}` 关闭；
  - Responses 格式用 `{"reasoning": {"effort": "none"}}` 关闭。
  - 📏 对一个简单问题，开启思考会让首 token 晚约 0.65 s（0.93 s 变成 1.58 s）。问题越难，差距越大。
- **非思考模式下的采样参数**：`temperature` 生效；`top_p` 固定为 1.0。
- **应用层工具**：
  - `deepseek_web_search`：关闭思考时是普通搜索，开启思考时是"深度思考搜索"；
  - `deepseek_read_url`；
  - `safety_check: {"level": "high"}`：面向公众提供服务时官方要求开启，命中风险时返回 `stop_reason: "refusal"`；
  - `title`。
  - 声明这些工具后，`temperature`、`top_p`、`stop` 等参数由服务端决定，请求里的值会被忽略。
- **价格**（元 / 百万 tokens）：

  | | 空闲时段 | 高峰时段 |
  |---|---|---|
  | 输入，缓存命中 | 0.02 | 0.04 |
  | 输入，缓存未命中 | 1 | 2 |
  | 输出 | 4 | 8 |

  高峰时段是北京时间工作日 9:00–12:00 和 14:00–18:00。
- **搜索和读链接**：每千次 5 元。

### 2.5 📏 实测延迟汇总

| 环节 | 本测试机实测（往返约 0.38 s） | 估算的服务端耗时（减 1 次往返） |
|---|---|---|
| 网络往返基线（`GET /models`） | 中位数 404 ms（359–792） | — |
| LLM 首 token，关闭思考，5 次 | 中位数 **930 ms**（578–1053） | 约 0.5 s |
| LLM 首 token，开启思考，2 次 | 1520 / 1644 ms | 约 1.2 s |
| LLM 回答 50 字的总耗时 | 中位数 1347 ms | 约 0.95 s |
| TTS HTTP 首包，3 / 10 / 17 / 34 / 108 字 | 约 **1.2–1.4 s**，**与文本长度无关** | 约 0.9–1.0 s |
| TTS WebSocket 建连 | 0.9–1.5 s | 经代理，参考价值有限 |
| ASR，2 s 语音，24 kHz WAV（99 KiB） | 中位数 1.06–1.2 s | 约 0.7–0.8 s |
| ASR，2 s 语音，16 kHz WAV（66 KiB） | 中位数 **1.04 s** | 约 0.65 s |
| ASR，7 s 语音，24 kHz WAV（331 KiB） | 中位数 1.69 s | 约 1.3 s |
| **端到端：用户文本 → 第一段音频**（3 个问题） | 1.99 / 2.32 / 2.52 s，**中位数 2.3 s**，播放无断档 | 约 1.6 s |

- ASR 耗时随语音长度增长，大约每多 1 s 语音多 0.1 s。上传体积也有影响，所以用 16 kHz 或 Ogg Opus。
- TTS 首包是**固定成本**，与文本长短无关。所以第一个分句只要语义完整，越早送出越好。

### 2.6 这些约束对实时对话的影响

| 约束 | 影响 | 对策 |
|---|---|---|
| ASR 非流式，短句也要约 0.7 s | 说完前没有任何文字；判停后还要再等 ASR | 投机 ASR（静音 250 ms 就先发）；延迟敏感时加本地流式 ASR，例如 FunASR 或 Qwen3-ASR |
| TTS 首包约 1 s 且是固定成本 | 回答总要多等约 1 s | **预合成语气词立即播放**；第一个分句尽早送；或换低首包的 TTS |
| WebSocket 不做增量合成 | "边生成边合成"走不通 | **逐句 HTTP 请求，并发获取、按序播放**（实测无断档） |
| 没有字级时间戳 | 打断后不知道说到哪了 | 按已播放时长和已合成字符数的比例估算 |
| 只有 2 个音色，不能调语速 | 个性化有限 | 需要更多音色时换 TTS |
| 汉字按 2 字符计费 | 回答越长越贵 | 提示词限制长度，一般 2–3 句 |
| 账号级 50 个并发请求（ASR 和 TTS 分别计） | 每通电话同一时刻会占 1–3 个 TTS 请求 | 同时通话数大约不超过 15–25 路；超过需要联系官方或分账号 |

---

## 3. 实时对话 Agent 设计

### 3.1 总体架构

```
 客户端（浏览器 / App）                 Agent Server（Python asyncio / Pipecat）                    DeepSeek API
┌───────────────────┐  WebRTC 或   ┌──────────────────────────────────────────────────────────┐
│ 麦克风 16 kHz PCM   │  WebSocket   │  ① VAD（Silero / TEN）──▶ ② 判停（静音阈值 + Smart Turn）  │
│ AEC/NS/AGC 开启    │ ───────────▶ │        │ 用户开口 = 打断信号             │ 语音段 → 16k WAV │ ── POST /v1/tools/asr ──▶
│                   │              │        ▼                                 ▼                 │ ◀──────── {text} ────────
│                   │              │  ⑥ 对话状态机 ◀──────────────── ③ ASR（含投机识别）        │
│ 播放队列 24 kHz     │              │   (IDLE/LISTENING/THINKING/SPEAKING)                     │ ── POST /chat/completions ─▶
│ 支持 clear 指令     │ ◀─────────── │        │   └─ 判停后立即播放预合成语气词（"嗯，"）          │   （stream，thinking 关闭）
│ 回报已播放毫秒数     │   PCM 帧     │        ▼                                                 │ ◀──────── delta ─────────
└───────────────────┘              │  ④ 前台 LLM（deepseek-flash）── 清洗 / 分句 ────────────▶   │ ── POST /v1/tools/tts ×N ─▶
                                   │        │ delegate_task(objective)   ⑤ 逐句并发 TTS，按序播放 │   （每句一个请求，keep-alive）
                                   │        ▼                                                 │ ◀──── PCM 24 kHz 流 ─────
                                   │  ⑦ 后台任务 Worker（flash + 思考 + 搜索 / ACP 编码 Agent）   │
                                   │        └── 结果入队，在安全窗口播报                         │
                                   └──────────────────────────────────────────────────────────┘
```

### 3.2 一轮对话的时序和延迟预算

```
用户 ████████████████▏(说完)
VAD                  ├ 静音 250 ms ┤→ 投机 ASR（16k WAV）──── 约 0.65–1.0 s ────┤
判停                 ├──── 600 ms 或 Smart Turn 确认 ────┤→ 立即播放语气词"嗯，"（0 延迟，掩盖等待）
LLM                                                                       ├─ 首 token 约 0.5–0.9 s ─┤ 首个分句
TTS（每句一个 HTTP 请求）                                                                            ├─ 首包约 1.0–1.4 s ─┤
播放                                                                                                                    ▶ 正式回答
```

| 环节 | 本测试机 | 国内机房（估算） | 优化手段 |
|---|---|---|---|
| 判停等待 | 600 ms | 600 ms | Smart Turn 降到约 300 ms；投机 ASR 与这段等待重叠 |
| ASR | 约 1.05 s | 约 0.7 s | 投机 ASR；16 kHz 或 Opus；加本地流式 ASR |
| LLM 首 token 加首个分句 | 约 0.6–1.2 s | 约 0.4–0.8 s | 关闭思考；系统提示词保持不变以命中缓存；控制历史长度；投机 LLM |
| TTS 首包 | 约 1.3 s | 约 1.0 s | 第一个分句尽早送；**预合成语气词**；或换低首包 TTS |
| 播放缓冲 | 约 0.1 s | 约 0.1 s | |
| **合计：说完 → 正式回答** | **约 3.7–4.3 s** | **约 2.8 s** | 投机 ASR 能省 0.35 s；投机 LLM 和投机 TTS 再省 0.5 s 左右；**全部用上约 2–2.3 s** |
| **说完 → 听到声音**（含语气词） | **约 0.6 s** | **约 0.6 s** | 语气词并不能代替回答，但能明显降低等待感 |

**语气词预合成**：服务启动时，用同一个音色（stella）调 DeepSeek TTS 合成一批短句并缓存成 PCM，例如"嗯，""好的，""我想想，""稍等，"。判停确认后立即播放一条，按场景随机选，避免机械感。这样运行时没有额外成本，音色也和正式回答一致。之后把这个语气词作为 assistant 的前缀写进历史，或者在提示词里告诉 LLM"你已经说过'嗯'了，不要再重复"。

### 3.3 关键模块设计

#### ① 音频采集和传输

- 浏览器端：`getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true}})`。**回声消除是打断功能的前提**，否则机器人自己的声音会被当成用户在说话。
- 传输：
  - MVP 用 WebSocket，上行 16 kHz PCM，每 20 ms 一帧；下行 24 kHz PCM。
  - 上线换 WebRTC（Pipecat SmallWebRTC 或 LiveKit）。
- 服务端部署在**离 `api.deepseek.com` 近的国内机房**。本次实测中，网络往返约占总延迟的三分之一。

#### ② VAD 和判停

- VAD：Silero VAD（Pipecat 内置）或 TEN VAD。静音约 250 ms 时触发投机 ASR。
- 判停：
  - 只看静音时，阈值设 600–800 ms；
  - 加 **Smart Turn v3**（Pipecat 内置本地模型，只看音频）可以把等待压到约 300 ms，同时不容易在用户思考的停顿中抢话。
- DeepSeek ASR 没有中间结果，所以文本语义判停只能等投机 ASR 返回后用规则判断，例如句尾是"然后""就是""那个"时再等一下。也可以加一个本地流式 ASR。

#### ③ ASR：分段加投机识别

```
VAD 静音 ≥250 ms   → 立即对当前音频段发起 ASR（投机）
确认本轮结束        → 投机结果已返回且之后没有新语音，直接使用
用户在确认前又开口  → 丢弃投机结果，音频段继续累积
```

- 上传 **16 kHz 单声道 WAV**（2 s 语音约 64 KiB）。实测比 24 kHz 快约 150 ms。也可以用 Ogg Opus 进一步缩小体积。
- 结果为空或只是语气词时，不进入 LLM。单段超过 60 s 就强制切段。
- 没有热词参数。专有名词识别不准时，在 LLM 提示词里给出"可能的同音误识别词表"，让 LLM 自行纠正。

#### ④ LLM：deepseek-flash

- 参数：`model="deepseek-flash"`、`stream=true`、`thinking={"type":"disabled"}`，`max_tokens` 控制在 300 左右。
- 系统提示词要点：
  - 用口语短句，一般 2–3 句；
  - 不使用 Markdown、列表、代码或表格；
  - 数字和单位写成方便朗读的形式；
  - 先说结论；
  - 输入来自语音识别，可能有同音错字，要按语境理解。
- 上下文：保留最近 N 轮，更早的对话异步压缩成摘要。系统提示词保持不变，便于命中缓存（缓存命中的输入价格只有未命中的 1/50）。
- 投机 LLM（第二阶段）：投机 ASR 一返回就开始请求 LLM，甚至把第一个分句也送去 TTS，但**音频先缓存不播**。本轮确认结束后再放行；用户又开口就取消。被浪费的成本每次不到 0.005 元。

#### ⑤ TTS：逐句 HTTP 请求，并发获取，按序播放

1. 文本先清洗（去掉 Markdown、链接、表情）。HTTP 接口默认 `filter_markdown=true`，可以兜底。
2. 切分：第一个分句到逗号、且不少于 6 个字就送出；之后按句号、问号、感叹号或分号切。
3. **每个分句一出来就立即发 HTTP 请求**（`format=pcm`，复用 keep-alive 连接），不用等上一句播完。音频按分句顺序进入播放队列。
   - 📏 实测 3 个问题都**没有播放断档**：第一句播放期间，后面几句的音频早已到达。
4. 每通电话同一时刻约占 1–3 个 TTS 并发，注意账号级 50 个并发的上限。
5. 记录每个分句的字符数和音频时长，供打断时估算截断位置。
6. 失败处理：遇到 429 或 503 就退避重试该分句；响应提前结束视为失败；400 `Content Exists Risk` 跳过该句，并在日志中告警。

#### ⑥ 打断（barge-in）和上下文截断

1. 状态为 SPEAKING 时，VAD 检测到用户连续说话 ≥ 300 ms，视为打断。阈值要比空闲时更高，避免把"嗯""对"这类附和误判成打断。
2. 打断时立即执行：
   - 取消 LLM 流；
   - 取消所有进行中的 TTS 请求；
   - 通知客户端 `clear` 播放队列。
3. 客户端回报已播放的毫秒数。服务端按分句的音频时长定位到"说到第几句的百分之几"，把 assistant 消息截断成"……（被用户打断）"再写入历史。

#### ⑦ 前台快答 + 后台委派（借鉴 qwen-audio-agent）

- 前台 LLM 只带轻量工具：
  - `delegate_task(objective)`：把用户意图改写成自包含的指令，**立即返回**；
  - `task_status()` 和 `cancel_task(id)`：直接读任务表。
- 委派的同时，前台先说一句"好，我去查一下，你可以继续说"。
- 后台 Worker：
  - 普通任务交给 `deepseek-flash` 开启思考，加上 `deepseek_web_search` 和 `deepseek_read_url`，也就是"深度思考搜索"；
  - 编码类任务通过 ACP 交给 DeepSeek Harness 或 Claude Code 等。
- 结果入队，等到"用户没在说、机器人也没在说"的窗口，由前台 LLM 把结果转述成口语后播报。
- 打断只推迟播报，不取消任务。权限确认必须来自用户当前这一轮的明确表态。

### 3.4 核心代码（2026-11-16 已用真实接口跑通 📏）

完整代码在 [`bench/pipeline.py`](../bench/pipeline.py)，端到端测试在 [`bench/e2e_probe.py`](../bench/e2e_probe.py)。下面是核心部分：

```python
import asyncio, io, json, os, wave
import httpx

KEY = os.environ["DEEPSEEK_API_KEY"]
API = "https://api.deepseek.com"
AUTH = {"Authorization": f"Bearer {KEY}"}

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
        await speak(http, chunk_for_tts(tee()), play)   # chunk_for_tts 见 bench/pipeline.py
    finally:                      # 被打断时这里只记下已生成的部分；按已播放时长截断见 3.3 节 ⑥
        history.append({"role": "assistant", "content": "".join(reply)})
```

📏 端到端测试输出（本测试机）：

```
Q=周末想在家附近放松一下，有什么建议？
  first_chunk_ready=607ms first_audio=1986ms audio=14.2s playback_gaps=0ms
  chunks: '可以睡到自然醒，' | '然后去附近公园散散步，晒晒太阳，顺便听点喜欢的音乐。' | '要是想更放松，……'
Q=怎么才能睡得更好？            first_chunk_ready=846ms  first_audio=2315ms audio=10.1s playback_gaps=0ms
Q=给我讲个简短的冷知识。         first_chunk_ready=1177ms first_audio=2523ms audio=10.3s playback_gaps=0ms
```

### 3.5 用 Pipecat 落地（推荐）

| 需要 | Pipecat 1.16.0 现成组件 ✅ | 要做的事 |
|---|---|---|
| LLM | `pipecat.services.deepseek.DeepSeekLLMService`（默认 `deepseek-flash`，默认关闭思考，`base_url` 为 `https://api.deepseek.com/v1`） | 直接用 |
| ASR | `SegmentedSTTService`：VAD 停止后把整段音频**封装成 WAV** 交给 `run_stt(audio)`，后台按顺序转写，尾部自动补静音；可以覆盖 `segment_sample_rate` 固定为 16000 | 写 `DeepSeekSTTService`：在 `run_stt` 里 POST multipart，yield `TranscriptionFrame`。参考 `whisper/base_stt.py` |
| TTS | `TTSService` 基类，设置 `text_aggregation_mode=TextAggregationMode.SENTENCE` 后**每句调用一次 `run_tts(text)`** | 写 `DeepSeekTTSService(TTSService)`：在 `run_tts` 里流式 POST `/v1/tools/tts`（`format=pcm`），yield `TTSAudioRawFrame(sample_rate=24000)`。**不要用 WebSocket 基类**（见 2.3 节实测） |
| VAD 和判停 | `SileroVADAnalyzer`、`LocalSmartTurnAnalyzerV3`，通过 `LLMUserAggregatorParams(vad_analyzer=..., user_turn_strategies=...)` 配置 | 调参 |
| 传输 | `transports/smallwebrtc`、`websocket`、`livekit`、`daily` | 选一个 |
| 打断 | 框架内置 | 验证截断效果 |
| 语气词、投机 ASR/LLM | 没有现成组件 | 自定义 FrameProcessor（第二阶段） |

流水线示意：

```python
pipeline = Pipeline([
    transport.input(),
    stt,                         # DeepSeekSTTService（自定义，SegmentedSTTService）
    context_aggregator.user(),   # 内含 Silero VAD 和 Smart Turn 判停
    llm,                         # DeepSeekLLMService(model="deepseek-flash")
    tts,                         # DeepSeekTTSService（自定义，TTSService，按句聚合）
    transport.output(),
    context_aggregator.assistant(),
])
```

> 注意：Pipecat 的 `TTSService` 是串行调用 `run_tts` 的。但 DeepSeek TTS 合成速度约为实时的 7 倍，一句话的音频会远早于它播完就全部到达，所以下一句的请求实际上在上一句播放期间就已经发出。
> 如果实测出现句间停顿，可以在 `run_tts` 外层自己做"预取下一句"。

### 3.6 其他路线

- **B. 自研**（FastAPI + WebSocket + `bench/pipeline.py`）：大约 800–1200 行，完全可控，便于试验投机策略。代价是 VAD、打断、抖动缓冲都要自己处理。
- **C. 接入 qwen-audio-agent**：把 B 包装成兼容 OpenAI Realtime 的 WebSocket，作为它的自定义 Provider。这样能直接复用它的后台 Agent 委派、权限管理和多端 UI。
- **D. 混合，降延迟**：
  - DeepSeek ASR + `deepseek-flash` + **低首包流式 TTS**（CosyVoice 2/3 自部署宣称约 150 ms，或其他云端实时 TTS），预计能把"说完到正式回答"压到约 1.5–2 s；
  - 再把 ASR 换成本地流式（FunASR 或 Qwen3-ASR），可以到约 1–1.5 s。
- **E. 前台用端到端模型**：前台用 Qwen Audio 3.1 Realtime 或开源 Qwen-Omni，延迟 1 s 以内；DeepSeek 只做后台推理和搜索。

### 3.7 成本估算（官方价格 ✅，限时五折）

| 项 | 单价 | 10 分钟通话（用户说 4 分钟，机器人说 5 分钟，约 30 轮） |
|---|---|---|
| ASR | 0.39 元/小时 | 约 0.026 元（投机识别再加约 10%） |
| TTS | 1.5 元/万计费字符，汉字算 2 | 中文约 270 字/分钟，即 540 计费字符/分钟；5 分钟约 **0.405 元** |
| LLM（deepseek-flash） | 输入命中 0.02–0.04，未命中 1–2，输出 4–8 元/百万 tokens | 每轮约 2k 输入（大部分命中缓存）和 100 输出，30 轮约 0.03–0.08 元 |
| 语气词预合成 | 一次性 | 可忽略 |
| **合计** | | **约 0.47–0.52 元**；恢复原价后约 0.9 元 |

### 3.8 风险和待验证项

1. **TTS 首包约 1 s 且是固定成本，WebSocket 不做增量合成**。这是延迟的主要瓶颈。关注官方更新，用 `bench/tts_ws_probe.py` 定期复测。
2. ASR 不支持流式，没有中间结果，因此没有实时字幕，也做不了文本语义判停。
3. 账号级 50 并发（ASR 和 TTS 分别计）限制了同时通话数，规模化需要联系官方。
4. 只有 2 个音色，不能调语速和情感；客服、陪伴类场景可能不够用。
5. 限时五折结束后成本翻倍，TTS 是大头。
6. 面向公众提供服务时，官方要求开启 `safety_check` 并履行备案和 AI 标识义务。TTS 也会对文本做内容审核（400 `Content Exists Risk`）。
7. 数据会发到 DeepSeek 服务端，有隐私合规要求的场景需要评估。
8. 本文的延迟数据来自海外测试机，**部署前请在目标机房复测**。

---

## 4. 实施路线

| 阶段 | 目标 | 内容 |
|---|---|---|
| **P0 验证** ✅（已完成，见 `bench/`） | 摸清接口 | ASR、LLM 流式、TTS HTTP 和 WebSocket 都已跑通；测得延迟分布；确认 WebSocket 不做增量合成 |
| **P1 MVP**（约 1 周） | 能打断的网页语音对话 | Pipecat + SmallWebRTC + Silero VAD + 三个 DeepSeek 服务；语音专用提示词；**预合成语气词**；各环节埋点；**部署到国内机房复测** |
| **P2 降延迟** | 说完到正式回答，P50 ≤ 2 s | Smart Turn v3、投机 ASR、投机 LLM 和投机 TTS；按需换低首包 TTS 或本地流式 ASR（路线 D） |
| **P3 Agent 化** | 能办事 | 前台快答加后台委派（深度思考搜索、读链接、ACP 编码 Agent），任务表，安全窗口播报，权限确认 |
| **P4 生产化** | 稳定可扩展 | 并发池和 429 退避、多账号或扩容、电话接入、`safety_check`、监控、录音与转写存档（需用户同意） |

---

## 5. 参考来源

**DeepSeek 官方文档**（2026-11-16 读取 Markdown 版本）
- [TTS 指南](https://api-docs.deepseek.com/zh-cn/guides/tts/)、[TTS API](https://api-docs.deepseek.com/api/tts/)
- [ASR 指南](https://api-docs.deepseek.com/zh-cn/guides/asr/)、[ASR API](https://api-docs.deepseek.com/zh-cn/api/asr.md)
- [更新日志 2026-11-13](https://api-docs.deepseek.com/updates/)、[构建聊天机器人](https://api-docs.deepseek.com/guides/build_a_chatbot.md)、[思考模式](https://api-docs.deepseek.com/guides/thinking_mode.md)、[模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)

**DeepSeek 相关仓库和报道**
- [deepseek-ai/chat-demo](https://github.com/deepseek-ai/chat-demo)（读过 `src/services/api/speech.ts`、`src/services/speech/recorder.ts`、`readAloud.ts`、`client.ts`）
- [JimmyWangJimmy/deepseek-app-api-tutorial](https://github.com/JimmyWangJimmy/deepseek-app-api-tutorial)（第三方教程）
- 报道：[IT之家](https://www.ithome.com/1/026/960.htm)、[腾讯新闻](https://news.qq.com/rain/a/20261114A0509N00)、[V2EX](https://www.v2ex.com/t/1257402)、[知乎](https://zhuanlan.zhihu.com/p/2104730286609134468)

**阿里系**
- [QwenAudio/qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent)（README、Releases、`docs/architecture/deep-dive.md`、`docs/voice-frontends/custom-provider.md`、`speech-to-speech.md`）
- [QwenLM/Qwen-Live-Harness](https://github.com/QwenLM/Qwen-Live-Harness)
- Qwen-Audio-3.0-Realtime：[量子位](https://www.qbitai.com/2026/07/450250.html)、[雷峰网](https://www.leiphone.com/category/industrynews/Yg2RqalAz09LS6WJ.html)
- [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)、[Fun-ASR 技术报告](https://arxiv.org/html/2509.12508v4)、[Fun-Audio-Chat 论文](https://arxiv.org/html/2512.20156v4)

**框架和综述**
- Pipecat 1.16.0（PyPI wheel 源码：`services/stt_service.py`、`services/tts_service.py`、`services/deepseek/llm.py`、`audio/turn/smart_turn/`）
- 框架对比：[AssemblyAI](https://www.assemblyai.com/blog/orchestration-tools-ai-voice-agents)、[Plivo](https://www.plivo.com/blog/how-to-build-a-voice-ai-agent-livekit-pipecat-ten-or-native/)、[RoomKit](https://www.roomkit.live/blog/choosing-the-right-conversational-ai-framework/)
- [全双工语音大模型全景调研（2026-09）](https://alwayszhang.cn/posts/2026-09-10-%E5%85%A8%E5%8F%8C%E5%B7%A5%E8%AF%AD%E9%9F%B3%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%85%A8%E6%99%AF%E8%B0%83%E7%A0%94-13%E4%B8%AA%E7%BB%B4%E5%BA%A6%E7%9C%8B%E6%87%82Moshi-GPT-Live-Gemini-Live-SeedRealtime-Gander%E6%8A%80%E6%9C%AF%E8%B7%AF%E7%BA%BF/)、[NVIDIA VoiceChat 11B（声网）](https://www.shengwang.cn/blog/blogdetail/NVIDIA-voicechat/)
- 判停和流式 ASR：[arXiv 2609.04225](https://arxiv.org/abs/2609.04225)、[HF speech-to-speech](https://github.com/huggingface/speech-to-speech)
