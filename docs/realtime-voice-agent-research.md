# 实时语音对话 Agent：开源方案调研 + 基于 DeepSeek ASR/TTS 的设计

> 调研日期：2026-11-16。DeepSeek 语音接口 2026-11-13 才发布，文档还可能变动。
> 来源可信度标注：
> - ✅ 已核实：读过官方仓库源码或 README。
> - 🟡 二手：来自第三方教程或媒体，未与官方文档逐条比对。
> - ❓ 未知：没有公开数据，需要自己测。
>
> 本环境的网络策略拦截了 `api-docs.deepseek.com`，所以官方文档页没有直接读到。DeepSeek 部分主要依据官方开源前端
> [deepseek-ai/chat-demo](https://github.com/deepseek-ai/chat-demo) 的源码，以及第三方教程
> [deepseek-app-api-tutorial](https://github.com/JimmyWangJimmy/deepseek-app-api-tutorial)。

---

## 0. 结论速览

1. **你听说的阿里开源语音 Agent 是 `qwen-audio-agent`**（GitHub 组织是 `QwenAudio`，Apache-2.0，2026-07-28 开源，当前 v2.1.0，约 3.2k star）。
   它不是模型，而是一个"全双工语音前台 + 后台 Agent"的运行时。前台的语音由云端实时模型生成，默认是 `qwen-audio-3.1-realtime-plus`，走百炼/DashScope API。
   后台通过 ACP 协议把任务委派给 Qwen Code、OpenCode、Claude Code、Codex、DeepSeek Harness 等编码 Agent。
2. **DeepSeek 的语音接口只能做"级联式"实时对话**：
   - ASR 是**整段上传、非流式**，单次最长 120 秒、最大 8 MiB。
   - TTS 支持 **WebSocket 流式输入文本、流式返回 PCM**，适合接 LLM 的流式输出。
   - 所以链路必须是：本地 VAD 和判停 → 分段上传 ASR → `deepseek-flash` 流式生成（关闭思考）→ TTS 流式合成。
3. **延迟预期**：首字音频大约 1.0–2.0 秒（估算，需实测）。端到端实时模型一般在 0.3–0.8 秒。
   差距主要来自"等用户说完再上传 ASR"这一步。用**投机 ASR、投机 LLM、TTS 预连接**大约能省 300–600 ms。
4. **推荐落地路径**：用 **Pipecat 1.16**（BSD-2，Python）。它已经内置这几样：
   - `DeepSeekLLMService`，默认 `deepseek-flash`，默认关闭思考；
   - `SegmentedSTTService` 基类，专门对接"上传整段音频"类的 ASR；
   - `InterruptibleTTSService` 基类，用于没有字级时间戳、靠重连实现打断的 WebSocket TTS；
   - Silero VAD、Smart Turn 判停、WebRTC 和 WebSocket 传输。

   只需要自己写两个约 150 行的类：`DeepSeekSTTService` 和 `DeepSeekTTSService`。
   第二阶段再借鉴 qwen-audio-agent 的"前台快答 + 后台委派"模式，处理需要搜索或多步操作的任务。
5. **成本**：TTS 是主要开销。按限时五折价，机器人每说 1 分钟中文约 0.08 元；ASR 每分钟用户语音约 0.0065 元。LLM 的 token 费用另算。

---

## 1. 开源方案全景

### 1.1 两条技术路线

| | 级联式（ASR → LLM → TTS） | 端到端 / 全双工语音模型 |
|---|---|---|
| 代表 | Pipecat、LiveKit Agents、TEN + 任意 ASR/LLM/TTS | Qwen-Omni 系列、Fun-Audio-Chat、Kimi-Audio、Step-Audio 2、MiniCPM-o 4.5、Moshi、GLM-4-Voice |
| 首字延迟 | 约 0.7–2 s | 约 0.3–0.8 s |
| LLM 智力和工具调用 | 可以接最强的文本模型，工具生态成熟 | 受限于语音模型本身，工具调用能力参差 |
| 副语言（情绪、语气、笑声） | 在 ASR 这一步丢失 | 能保留 |
| 可控性和调试 | 每段都可以替换、记日志 | 黑盒程度更高 |
| **DeepSeek 适用性** | **✅ 唯一可行的方式** | DeepSeek 只能做后台的"慢思考" |

业界的常见做法是把两条路线组合起来：用端到端实时模型做前台快速对话，再把需要深度推理或工具调用的任务交给后台文本 Agent。qwen-audio-agent 走的就是这条路。

### 1.2 阿里系

| 项目 | 类型 | 开源情况 | 要点 |
|---|---|---|---|
| **[qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent)** ✅ | 语音 Agent 运行时（Node.js） | Apache-2.0 | 见下方详述 |
| [Qwen-Live-Harness](https://github.com/QwenLM/Qwen-Live-Harness) ✅ | 桌面 Agent（Electron + Node 守护进程） | Apache-2.0 | 用 Qwen Omni Realtime API "看、听、做"，内置记忆。后台委派给 Qwen Code、Qoder、Codex、Claude Code、Gemini CLI（ACP）。目前只支持 macOS，需要 DashScope Key |
| Qwen-Audio-3.0/3.1-Realtime 🟡 | 实时语音大模型 | **只提供 API**，没有找到开放权重的说明 | 3.0 版本于 2026-07 发布，有 Plus 和 Flash 两档。能自主调用工具，内置"多模态感知双工控制"子模型来判断对话节奏 |
| Qwen3-Omni / Qwen-Omni 系列 🟡 | 端到端全模态模型 | 开源权重 | 可以本地部署做语音对话 |
| Fun-Audio-Chat 🟡 | 端到端语音对话模型 | 开源（论文 arXiv 2512.20156） | 用 5 Hz 低帧率共享 LLM 主干、25 Hz 语音头，以降低算力开销 |
| Qwen3-ASR 🟡 | ASR | Apache-2.0，0.6B 和 1.7B 两档 | 支持多语种和方言；**流式推理目前只在 vLLM 后端可用** |
| FunASR / Fun-ASR-Nano-2512 🟡 | ASR 工具包 | 开源 | 有 paraformer 流式模型和 2-pass 方案。2026-05 加入 vLLM 加速，并提供带 VAD 和说话人分离的流式 WebSocket 服务。Pipecat 已内置 `FunASRSTTService` |
| SenseVoice 🟡 | ASR 和情感/事件识别 | 开源 | 非自回归，速度很快 |
| CosyVoice 2/3 🟡 | TTS | 开源 | 支持流式；第三方称流式首包约 150 ms |
| Qwen3-TTS 🟡 | TTS | 开源，0.6B 和 1.7B | 宣称输入一个字就能输出首个音频包 |

#### qwen-audio-agent 详述 ✅

- **架构**有三部分：
  1. 前台 Agent：实时语音模型，加上指令、上下文和轻量工具；
  2. 编排运行时：管理任务、权限、会话和事件；
  3. 后台 Agent：一个通过 ACP 启动的子进程，在用户环境里干活。

  Gateway 以服务形式托管这三部分，提供 Web UI、终端 TUI 和桌面悬浮球。
- **非阻塞委派**：前台调用 `spawn_thinking(objective)` 后**立即返回**，所以用户可以继续说话。任务进入有上限的队列，默认全局只有 1 个活跃槽位。
  结果出来后，要等到"安全的双工插话窗口"才播报。`objective` 由前台模型把用户的话**改写成一条自包含的指令**，而不是直接传 ASR 原文。
- **前台工具**：`update_agent_task`、`get_agent_task_status`、`cancel_agent_task`、`respond_permission`，以及记忆、搜索等可选工具。
  查询状态、取消任务都直接读任务记录，不经过 LLM 重新推理。取消要等后台确认，不能只凭模型口头说"已取消"。
- **打断**：用户打断只会推迟播报，**不会取消已经提交的后台任务**。权限审批必须来自用户**当前这一轮**的明确表态。
- **可插拔的实时 Provider**：
  - 内置 Qwen Audio 3.x Realtime、GPT-Live/OpenAI Realtime、Gemini Live、Qwen3.5-Omni、Qwen3.8 Omni Flash、豆包 Seeduplex 3.0、StepAudio 3 Realtime、MiniCPM-o 4.5。
  - 还内置 **Hugging Face speech-to-speech**。这是一个本地级联方案（VAD + STT + LLM + TTS），对外暴露兼容 OpenAI Realtime 的 WebSocket，地址是 `ws://127.0.0.1:8765/v1/realtime`。
  - **前台没有 DeepSeek**。DeepSeek 只作为后台出现（DeepSeek Harness，原生 ACP）。
- **自定义 Provider 接口**：需要实现 `url()`、`headers()`、`buildSession()`、`buildSpeakResponse()`、`buildResultInjection()`、`inputSampleRate`、`outputSampleRate` 等方法。
- **安装**：

  ```bash
  npm i -g qwen-audio-agent
  ```

  然后运行 `qwenaudio config`，再启动 `qwenaudio`，最后打开 `qwenaudio webui` 或 `qwenaudio tui`。

### 1.3 编排框架（级联式）

| 框架 | 许可 | 语言 | 适合场景 |
|---|---|---|---|
| **[Pipecat](https://github.com/pipecat-ai/pipecat)** ✅（读过 1.16.0 源码） | BSD-2 | Python | 用帧流水线组织处理器，集成的服务最多，便于替换供应商。**已内置 DeepSeek LLM、FunASR、Qwen LLM、Silero VAD、Smart Turn v2/v3**，传输层支持 SmallWebRTC、WebSocket、Daily、LiveKit |
| [LiveKit Agents](https://github.com/livekit/agents) | Apache-2.0 | Python 和 Node | 基于 WebRTC SFU，适合多人、电话、视频场景；自带判停模型和多 Agent 交接 |
| [TEN Framework](https://github.com/TEN-framework/ten-framework) | 需自行核实（各来源说法不一） | 多语言图编排 | 适合多模态和数字人。TEN VAD（MIT）和 TEN 判停模型可以单独拿来用 |
| [HF speech-to-speech](https://github.com/huggingface/speech-to-speech) | Apache-2.0 | Python | 纯本地级联，可作为 qwen-audio-agent 的 Provider |

### 1.4 其他端到端开源模型 🟡

- Kimi-Audio、Step-Audio 2、GLM-4-Voice、Moshi
- MiniCPM-o 4.5：全双工，但不支持工具调用
- NVIDIA PersonaPlex：基于 Moshi
- NVIDIA NemotronLabs VoiceChat 11B：号称首个支持实时工具调用的开源全双工模型，轮换延迟 448 ms

如果目标是"全本地、低延迟、离线运行"，应该从这一类模型里选，而不是用 DeepSeek 的云端接口。

### 1.5 DeepSeek 生态现状

- [deepseek-ai/chat-demo](https://github.com/deepseek-ai/chat-demo) ✅：官方前端，MIT，纯静态页面。语音功能只有"按住录音后识别"和"整段朗读"，**不是实时通话**。
- [hangi7890/deepseek-live](https://github.com/hangi7890/deepseek-live) 🟡：MIT，支持打断，但用的是浏览器自带的 STT，只把文本发给 DeepSeek。
- [dsh-voice-talk](https://github.com/duoduoqian708/dsh-voice-talk) 和 [dsh-voice-call](https://github.com/biliye/dsh-voice-call) 🟡：DeepSeek Harness 的语音插件，TTS 用千问、讯飞或系统语音。
- 目前**没有看到**用 DeepSeek 官方 ASR + TTS 做的成熟实时对话开源项目，这里有机会。

### 1.6 选型建议

| 你的目标 | 推荐 |
|---|---|
| 用 DeepSeek 全家桶做实时语音助手（本文主题） | Pipecat 加两个自定义 DeepSeek 服务（见第 3 节） |
| 给编码 Agent 加"打电话式"语音前台 | 直接用 qwen-audio-agent，后台选 DeepSeek Harness |
| 追求最低延迟和情感表现 | 前台用端到端实时模型（Qwen Audio 3.1 Realtime 或开源 Qwen-Omni），DeepSeek 做后台 |
| 完全离线 | FunASR 或 Qwen3-ASR、本地 LLM、CosyVoice 或 Qwen3-TTS，用 Pipecat 编排；或者直接用端到端开源模型 |

---

## 2. DeepSeek 语音相关接口要点（2026-11-13 发布）

### 2.1 ASR：`POST https://api.deepseek.com/v1/tools/asr`

| 项 | 内容 | 来源 |
|---|---|---|
| 鉴权 | `Authorization: Bearer <key>` | ✅ chat-demo |
| 请求体 | `multipart/form-data`，只有一个字段 `file`，没有 model 或语种参数 | ✅ chat-demo |
| 格式 | 16-bit PCM WAV 或 Ogg Opus，根据文件内容判断。WAV 支持 8k、16k、22.05k、24k、32k、44.1k、48k 采样率；双声道只识别左声道 | 🟡 教程 |
| 限制 | 最大 8 MiB，最长 120 s，上传须在 10 分钟内完成 | 🟡 教程（chat-demo 代码里也写了 `MAX_SECONDS = 120`） |
| 流式 | **不支持**，一次提交整段，返回完整文本 | 🟡 官方文档摘要和教程 |
| 返回 | `{ request_id, text, duration_ms }` | 🟡（chat-demo 只读取 `text`） |
| 语种 | 21 种，自动识别 | 🟡 |
| 价格 | 0.39 元/小时（限时五折，原价 0.78），按秒计费，不足 1 秒按 1 秒算 | 🟡 |
| 并发 | 约 50 | 🟡 只出现在教程的检查清单里，未见官方说明 |

chat-demo 的录音做法 ✅：用 `getUserMedia`，开启 `echoCancellation` 和 `noiseSuppression`，在 16 kHz 的 AudioContext 里采集 PCM，在前端编码成 16 kHz 单声道 WAV 再上传。

### 2.2 TTS（HTTP）：`POST https://api.deepseek.com/v1/tools/tts`

- 请求体 ✅：`{"text": "...", "voice_id": "stella" | "tide", "format": "pcm" | "wav", "filter_markdown": true}`
  - stella 是女声，tide 是男声；
  - `filter_markdown` 让服务端去掉代码块、表格、引用标记和长公式。
- 返回 ✅：边生成边流式返回。`pcm` 格式是 **24 kHz、16-bit 有符号、单声道、小端**。
- 限制 🟡：每次请求最多 10,000 个计费字符、请求体最大 64 KiB、音频最长 30 分钟；**不能调语速**。
- 价格 🟡：1.5 元 / 万字符（限时五折，原价 3.0）。**一个汉字算 2 个字符**，字母、数字、空格各算 1 个。

### 2.3 TTS（WebSocket 流式输入）：`wss://api.deepseek.com/v1/tools/tts?voice_id=stella&format=pcm`

🟡 以下来自第三方教程的示例脚本和搜索到的官方文档摘要。**官方 chat-demo 没有使用 WebSocket**，接入前请对照官方文档确认。

- **鉴权**：请求头 `Authorization: Bearer <key>`。官方文档摘要还提到可以用子协议 `Sec-WebSocket-Protocol: tts, bearer.<key>`，供浏览器直连使用。不过**不要把 Key 暴露给浏览器**，应该由服务端代理。
- **客户端发送**：
  - 每段文本发一个 JSON 帧：`{"event": "text", "text": "<片段>"}`；
  - 文本发完后发 `{"event": "finish_text"}`。
- **服务端返回**：
  - 二进制帧是 24 kHz s16le 单声道 PCM；
  - 文本帧是 JSON，带 `code` 字段。`code == "success"` 表示结束，其他值表示出错，此时带 `message`。
- **格式**：`pcm` 或 `opus`。默认值在官方文档摘要和 API 参考之间说法不一致，建议显式传参。
- **保活和超时**：服务端每 25 秒发一次 ping；**60 秒内没有收到文本就断开**；单个会话最长 30 分钟或 10,000 计费字符。
- **待确认**：WebSocket 模式是否支持 `filter_markdown`（未确认），以及服务端怎么根据收到的文本片段断句。

### 2.4 LLM：`deepseek-flash`

- 应用层工具（联网搜索、读链接、文件、安全审核）目前**只支持 `deepseek-flash`**（V4.1-Flash）。
- 有两种调用方式：
  - OpenAI 兼容：`https://api.deepseek.com/v1/chat/completions`。✅ Pipecat 内置的 `DeepSeekLLMService` 就用这个，默认 `model="deepseek-flash"`，并通过 `extra_body` 传 `{"thinking": {"type": "disabled"}}`。
  - Anthropic 兼容：`https://api.deepseek.com/anthropic/v1/messages`，✅ chat-demo 用的是这个。
- **实时语音必须关闭思考**。V4 模型默认会先推理，这会明显推迟第一个 token。
- 内置工具 🟡：`deepseek_web_search`、`deepseek_read_url`。**声明搜索工具后会默认开启思考**，需要显式传 `thinking: disabled`。开启搜索、读链接等功能后，temperature 和 top_p 会被忽略。

### 2.5 这些约束对实时对话意味着什么

| 约束 | 影响 | 对策 |
|---|---|---|
| ASR 非流式 | 用户说完之前拿不到任何文字，无法用中间结果提前判停或实时字幕 | 本地 VAD 加判停模型；投机 ASR；需要实时字幕或更强判停时，可以加一个本地流式 ASR 只出中间结果 |
| ASR 单段 ≤120 s | 长段独白要切分 | VAD 分段天然满足；超过 60 s 就强制切段 |
| TTS 支持流式输入 | ✅ LLM 的增量文本可以直接送进去，不用等整句 | 每轮开一个 WebSocket 会话，边生成边送 |
| TTS 没有字级时间戳 | 打断后不知道"说到哪了" | 按已播放音频时长和已合成字符数的比例估算截断位置 |
| TTS 60 s 无文本就断开 | 不能在用户开口时就预连 | 在判停触发时预连，提前约 300–700 ms |
| TTS 计费：汉字算 2 字符 | 回答越长越贵 | 提示词要求口语化、简短，3 句以内 |

---

## 3. 实时对话 Agent 设计

### 3.1 总体架构

```
 客户端（浏览器 / App）                 Agent Server（Python asyncio / Pipecat）                   DeepSeek API
┌───────────────────┐  WebRTC 或   ┌─────────────────────────────────────────────────────────┐
│ 麦克风 16 kHz PCM   │  WebSocket   │  ① VAD（Silero / TEN）──▶ ② 判停（静音阈值 + Smart Turn） │
│ AEC/NS/AGC 开启    │ ───────────▶ │        │ 用户开口 = 打断信号             │ 语音段 → WAV    │
│                   │              │        ▼                                 ▼                │ ── POST /v1/tools/asr ──▶
│                   │              │  ⑥ 对话状态机 ◀──────────────── ③ ASR（含投机识别）       │ ◀──────── {text} ────────
│ 播放队列 24 kHz     │              │   (IDLE/LISTENING/THINKING/SPEAKING)                    │
│ 支持 clear 指令     │ ◀─────────── │        │                                                │ ── /v1/chat/completions ─▶
│ 回报已播放毫秒数     │   PCM 帧     │        ▼                                                │    （stream，thinking 关闭）
└───────────────────┘              │  ④ 前台 LLM（deepseek-flash）── 文本清洗 / 分片 ────────▶  │ ◀──────── delta ─────────
                                   │        │ delegate_task(objective)                        │
                                   │        ▼                                                │ ══ wss /v1/tools/tts ══▶
                                   │  ⑦ 后台任务 Worker（flash + 思考 + 搜索 / ACP 编码 Agent）  │ ◀═══ PCM 24 kHz ═══════
                                   │        └── 结果入队，在安全窗口播报 ──▶ ⑤ TTS 会话           │
                                   └─────────────────────────────────────────────────────────┘
```

### 3.2 一轮对话的时序和延迟预算

```
用户 ████████████████▏(说完)
VAD                  ├─ 静音 250ms ─┤→ 投机 ASR 发起（上传约 100 KB WAV）
                     ├──── 静音 600ms 或 Smart Turn 判定"已说完" ────┤→ 确认本轮结束
ASR                                  ├──── ASR 耗时 ❓ ────┤
LLM                                                       ├─ TTFT ❓ ─┤ 首个短语
TTS                  （判停时已预建 WebSocket）                         ├─ 首包 ❓ ─┤
播放                                                                               ├ 缓冲约 100ms ┤▶ 用户听到
```

| 环节 | 估计（需实测） | 优化手段 |
|---|---|---|
| 判停等待 | 500–800 ms | Smart Turn 语义判停，把静音阈值降到约 300 ms；投机 ASR 和投机 LLM 与这段等待重叠 |
| ASR（短句上传加识别） | ❓ 估计 200–600 ms | 投机发起；16 kHz 单声道 WAV（5 秒约 160 KB），带宽紧张时改用 Ogg Opus |
| LLM 首 token（flash，关闭思考） | ❓ 估计 300–800 ms | 关闭思考；系统提示词保持稳定，利于前缀缓存；控制历史长度 |
| 首个可送 TTS 的片段 | 50–200 ms | 第一句遇到逗号就送，或者攒满约 8 个字就送 |
| TTS 首包 | ❓ 估计 200–500 ms | 判停时预建 WebSocket；用 `pcm` 省去容器开销 |
| 传输和播放缓冲 | 100–150 ms | 播放端的 jitter buffer 控制在 80–120 ms |
| **合计** | **不优化约 1.5–2.5 s；优化后约 1.0–1.6 s** | |

**第一件事是埋点实测**。每轮记录以下时间戳，然后看 P50 和 P90：

- `vad_stop`、`asr_sent`、`asr_done`、`llm_first_token`、`tts_first_text`、`tts_first_audio`、`client_playback_start`

注意 TTS 首包时间不等于用户等待时间，以客户端开始播放的时刻为准。Pipecat 的 metrics 自带 TTFB 统计。

### 3.3 关键模块设计

#### ① 音频采集和传输

- 浏览器端：`getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true}})`。**回声消除是打断功能能用的前提**，否则机器人自己的声音会被当成用户在说话。
- 传输：
  - MVP 用 WebSocket，上行 16 kHz PCM，每 20 ms 一帧；下行 24 kHz PCM。
  - 上线时换 WebRTC（Pipecat SmallWebRTC 或 LiveKit），获得更好的抗抖动、AEC 和弱网表现。
- 电话场景：接 Twilio、Vonage 或 SIP。注意 8 kHz 和 24 kHz 之间的重采样。

#### ② VAD 和判停

- VAD：Silero VAD（Pipecat 内置）或 TEN VAD（MIT，号称精度更高、开销更低）。参数参考：起说阈值约 200 ms，静音阈值 250–300 ms，后者用来触发投机 ASR。
- 判停：
  - 只看静音时，阈值设 600–800 ms；
  - 加上 **Smart Turn v3**（Pipecat 内置本地模型，直接看音频，不需要转写），可以把等待压到约 300 ms，同时不容易在用户思考的停顿中抢话。
- 由于 DeepSeek ASR 没有中间结果，**判停只能依靠声学信号**。如果想用"文本是否说完"做语义判停，可以：
  - 等投机 ASR 返回后用规则判断（例如句尾是"然后""就是""那个"，就再等一下）；
  - 或者加一个本地流式 ASR 专门出中间结果，例如 FunASR paraformer-streaming 或 Qwen3-ASR + vLLM。

#### ③ ASR：分段加投机识别

```
VAD 静音 ≥250 ms   → 立即对当前音频段发起 ASR（投机）
确认本轮结束        → 如果投机结果已返回且之后没有新语音，直接使用
用户在确认前又开口  → 丢弃投机结果，音频段继续累积
```

- 额外成本很低：按秒计费，被丢弃的请求每次也就几秒钱。
- 噪音或误触发：ASR 返回空文本或纯语气词时，不进入 LLM。
- 120 秒上限：单段超过 60 秒就强制切段，分段识别后拼接。
- DeepSeek ASR 没有热词或上下文参数（未见文档提及）。专有名词识别不准时，可以在 LLM 提示词里给出"可能的同音误识别词表"，让 LLM 自行纠正。

#### ④ LLM：deepseek-flash

- 参数：`model="deepseek-flash"`、`stream=True`、`thinking={"type": "disabled"}`，`max_tokens` 控制在 300 左右。
- 系统提示词要点（语音专用）：
  - 用口语短句，一般不超过 3 句；
  - 不使用 Markdown、列表、代码或表格；
  - 数字、单位、英文缩写写成方便朗读的形式；
  - 不确定时先给结论再补充；
  - 用户的输入来自语音识别，可能有同音错字，要按语境理解。
- 上下文：保留最近 N 轮，更早的对话异步压缩成摘要。系统提示词保持不变，以利于前缀缓存、降低 TTFT。
- 投机 LLM（可选，第二阶段）：投机 ASR 一返回就开始请求 LLM，但**输出先缓存不播**；本轮确认结束后再放行。用户又开口就取消。

#### ⑤ TTS：每轮一个 WebSocket 会话

- 判停触发时预连 `wss://.../v1/tools/tts?voice_id=stella&format=pcm`。不要在用户开口时就连，因为 60 秒没有文本会被断开。
- 文本送入策略：
  1. 先在客户端清洗，去掉 `* # \``、链接和表情（WebSocket 模式是否支持 `filter_markdown` 尚未确认）；
  2. **第一段尽快送出**：遇到第一个逗号或句号就送，或者攒满约 8 个字；
  3. 之后按句号、问号、感叹号或分号送，兼顾韵律；
  4. LLM 结束后发 `finish_text`，收到 `code == "success"` 后关闭连接。
- 记录每段文本送出的位置和收到的音频字节数，打断时用来估算截断位置。
- 并发：每个活跃通话同时只有 1 个 TTS 会话。教程称并发上限约 50，按"同时说话的通话数"规划容量。

#### ⑥ 打断（barge-in）和上下文截断

1. 状态为 SPEAKING 时，VAD 检测到用户连续说话 ≥ 300 ms，就视为打断。阈值要比空闲时更高，避免把"嗯""对"这类附和误判成打断。
2. 打断时立即执行：
   - 取消 LLM 流；
   - 关闭 TTS WebSocket（Pipecat 的 `InterruptibleTTSService` 就是"打断时重连"）；
   - 通知客户端 `clear` 播放队列。
3. 客户端回报本轮已播放的毫秒数，服务端估算已说出的文字：已播放时长 ÷ 已收到的音频总时长 × 已送出的字符数。然后把 assistant 消息截断成"……（被用户打断）"再写入历史，避免 LLM 以为自己已经说完了。
4. 如果被打断的那段话 ASR 结果为空或只是语气词，可以选择从断点"恢复播报"。这一项是可选的。

#### ⑦ 前台快答 + 后台委派（借鉴 qwen-audio-agent）

- 前台 LLM 只带轻量工具：
  - `delegate_task(objective)`：把用户意图改写成一条自包含的指令，**立即返回**；
  - `task_status()` 和 `cancel_task(id)`：直接读任务表，不经过 LLM 推理。
- 委派的同时，前台先说一句"好，我查一下，你可以继续说"。
- 后台 Worker：
  - 普通任务交给 `deepseek-flash` 开启思考，加上 `deepseek_web_search`、`deepseek_read_url`；
  - 编码类任务通过 ACP 交给 DeepSeek Harness 或 Claude Code 等。
- 结果入队，等到"用户没在说、机器人也没在说"的窗口，由前台 LLM 把结果**转述成口语**后播报（结果作为一条 system 或 tool 消息注入）。
- 打断只推迟播报，不取消任务。涉及权限的操作，必须由用户当前这一轮明确同意。

### 3.4 关键代码草图（框架无关，Python）

> 示意代码，未用真实 Key 跑过。WebSocket 协议细节以官方文档为准。

```python
import asyncio, io, json, os, wave
import httpx, websockets  # websockets >= 14

KEY = os.environ["DEEPSEEK_API_KEY"]
API = "https://api.deepseek.com"

def pcm_to_wav(pcm: bytes, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    return buf.getvalue()

async def asr(http: httpx.AsyncClient, pcm16k: bytes) -> str:
    r = await http.post(f"{API}/v1/tools/asr",
                        headers={"Authorization": f"Bearer {KEY}"},
                        files={"file": ("utt.wav", pcm_to_wav(pcm16k), "audio/wav")},
                        timeout=15)
    r.raise_for_status()
    return r.json()["text"].strip()

async def llm_stream(http: httpx.AsyncClient, messages):
    """OpenAI 兼容流式接口，逐个 yield 文本增量。"""
    body = {"model": "deepseek-flash", "messages": messages, "stream": True,
            "max_tokens": 300, "thinking": {"type": "disabled"}}
    async with http.stream("POST", f"{API}/v1/chat/completions", json=body,
                           headers={"Authorization": f"Bearer {KEY}"}, timeout=30) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
            if delta:
                yield delta

PUNCT_SOFT, PUNCT_HARD = "，,、：:", "。！？!?；;\n"

async def chunk_for_tts(deltas, first_min=8):
    """第一段尽快送出（遇到逗号或攒满 first_min 个字），之后按整句送。"""
    buf, first = "", True
    async for d in deltas:
        buf += d.replace("*", "").replace("#", "").replace("`", "")
        cut = max((buf.rfind(p) for p in (PUNCT_SOFT + PUNCT_HARD if first else PUNCT_HARD)), default=-1)
        if cut >= 0 or (first and len(buf) >= first_min):
            idx = cut + 1 if cut >= 0 else len(buf)
            yield buf[:idx]; buf = buf[idx:]; first = False
    if buf.strip():
        yield buf

async def tts_session(text_chunks, on_pcm24k, voice="stella"):
    """一轮回复对应一个 TTS WebSocket 会话：边送文本，边收 PCM。"""
    url = f"wss://api.deepseek.com/v1/tools/tts?voice_id={voice}&format=pcm"
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {KEY}"}) as ws:
        async def sender():
            async for piece in text_chunks:
                await ws.send(json.dumps({"event": "text", "text": piece}, ensure_ascii=False))
            await ws.send(json.dumps({"event": "finish_text"}))
        send_task = asyncio.create_task(sender())
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    await on_pcm24k(msg)          # 24 kHz s16le mono，转发给客户端播放
                else:
                    evt = json.loads(msg)
                    if evt.get("code") != "success":
                        raise RuntimeError(f"TTS error: {evt}")
                    break                         # 本轮合成结束
        finally:
            send_task.cancel()                    # 打断时外层直接 cancel 本协程即可

async def respond(http, history, user_text, send_audio):
    history.append({"role": "user", "content": user_text})
    reply = []
    async def tee():
        async for d in llm_stream(http, history):
            reply.append(d); yield d
    await tts_session(chunk_for_tts(tee()), send_audio)
    history.append({"role": "assistant", "content": "".join(reply)})
    # 被打断时，外层 cancel 这个 task，再按已播放时长截断 history[-1]
```

### 3.5 用 Pipecat 落地（推荐）

Pipecat 1.16.0 已经具备的组件 ✅：

| 需要 | Pipecat 现成组件 | 要做的事 |
|---|---|---|
| LLM | `pipecat.services.deepseek.DeepSeekLLMService`（默认 `deepseek-flash`，默认关闭思考） | 直接用 |
| ASR | `SegmentedSTTService` 基类。VAD 停止后把整段音频**封装成 WAV** 交给 `run_stt(audio)`，在后台按顺序转写，尾部自动补静音 | 写 `DeepSeekSTTService`：在 `run_stt` 里 POST multipart，yield `TranscriptionFrame`。可以参考 `whisper/base_stt.py` |
| TTS | `InterruptibleTTSService`（继承自 `WebsocketTTSService`），适用于没有字级时间戳、打断时靠重连的 WebSocket TTS | 写 `DeepSeekTTSService`：`_connect` 带上 voice_id 和 format 参数，`run_tts` 发 `{"event":"text"}`，flush 时发 `finish_text`，二进制帧推成 `TTSAudioRawFrame(sample_rate=24000)`。注意**每轮结束后需要重连** |
| VAD 和判停 | `SileroVADAnalyzer`，`smart_turn/local_smart_turn_v3.py`，通过 `LLMUserAggregatorParams(vad_analyzer=..., user_turn_strategies=...)` 配置 | 调参 |
| 传输 | `transports/smallwebrtc`、`websocket`、`livekit`、`daily` | 选一个 |
| 打断和上下文截断 | 框架内置 | 验证截断效果 |

流水线示意（具体以 Pipecat 1.16 的示例为准）：

```python
pipeline = Pipeline([
    transport.input(),
    stt,                         # DeepSeekSTTService（自定义，SegmentedSTTService）
    context_aggregator.user(),   # 内含 Silero VAD 和 Smart Turn 判停
    llm,                         # DeepSeekLLMService(model="deepseek-flash")
    tts,                         # DeepSeekTTSService（自定义，InterruptibleTTSService）
    transport.output(),
    context_aggregator.assistant(),
])
```

### 3.6 其他可选路线

- **B. 自研**（FastAPI + WebSocket + 3.4 节的代码）：大约 800–1200 行，完全可控，便于实验各种投机策略。代价是 VAD、打断、抖动缓冲、WebRTC 都要自己处理。
- **C. 接入 qwen-audio-agent**：把 B 包装成兼容 OpenAI Realtime 的 WebSocket（参考它的 HF speech-to-speech Provider），或者写一个自定义 Provider。这样能直接复用它的后台 Agent 委派、权限、Web/TUI/桌面端。适合"给编码 Agent 加语音"。
- **D. 混合**：前台用 Qwen Audio 3.1 Realtime 或开源 Qwen-Omni 获得亚秒级延迟和情感表达，DeepSeek 只做后台推理和搜索。这条路不用 DeepSeek 的 ASR 和 TTS。

### 3.7 成本估算（限时五折价，🟡）

| 项 | 单价 | 10 分钟通话（用户说 4 分钟，机器人说 5 分钟） |
|---|---|---|
| ASR | 0.39 元/小时 | 4 分钟约 0.026 元（投机识别再加约 10%） |
| TTS | 1.5 元/万计费字符，汉字算 2 | 中文约 270 字/分钟，即 540 计费字符/分钟；5 分钟约 0.405 元 |
| LLM | deepseek-flash 的 token 单价 | 另计（每轮都带上下文，注意前缀缓存） |
| **合计（不含 LLM）** | | **约 0.43 元**；恢复原价后约 0.86 元 |

### 3.8 风险和待验证项

1. ❓ TTS WebSocket 的结束事件格式、`filter_markdown` 是否可用、对短片段的断句效果。需要对照官方文档，并用真实 Key 跑通。
2. ❓ ASR 和 TTS 的实际延迟（P50 和 P90）。这决定投机策略值不值得做。
3. 🟡 并发和限流（ASR 和 TTS 各约 50）。需要在控制台确认。
4. ASR 不支持流式，所以没有实时字幕、不能做语义判停。必要时加本地流式 ASR 作为补充。
5. TTS 只有 2 个音色，不能调语速和情感，对客服、陪伴类场景可能不够用。可以换 CosyVoice 或 Qwen3-TTS。
6. 限时五折结束后成本翻倍，TTS 是大头。
7. 数据会发到 DeepSeek 服务端，涉及隐私合规的场景需要评估。

---

## 4. 实施路线

| 阶段 | 目标 | 内容 |
|---|---|---|
| **P0 验证**（1–2 天） | 摸清接口 | 用 3.4 节的脚本跑通 ASR、LLM 流式、TTS WebSocket；测 20 条样本的延迟分布；确认 WebSocket 协议细节 |
| **P1 MVP**（约 1 周） | 能打断的网页语音对话 | Pipecat + SmallWebRTC + Silero VAD + 两个自定义 DeepSeek 服务；语音专用提示词；埋点统计各环节 TTFB |
| **P2 降延迟** | P50 首字 ≤ 1.2 s | Smart Turn v3、投机 ASR 和投机 LLM、TTS 预连、首段快速送出、打断后的上下文截断 |
| **P3 Agent 化** | 能办事 | 前台快答加后台委派（搜索、读链接、ACP 编码 Agent），任务表，安全窗口播报，权限确认 |
| **P4 生产化** | 稳定可扩展 | 限流和重试、并发池、电话接入（SIP 或 Twilio）、监控、录音与转写存档（需用户同意） |

---

## 5. 参考来源

- DeepSeek 官方：[TTS 指南](https://api-docs.deepseek.com/guides/tts/)、[TTS API](https://api-docs.deepseek.com/api/tts/)、[ASR 指南](https://api-docs.deepseek.com/zh-cn/guides/asr/)、[ASR API](https://api-docs.deepseek.com/zh-cn/api/asr.md)、[更新日志](https://api-docs.deepseek.com/updates/)（本环境无法直接访问，内容来自搜索摘要）
- [deepseek-ai/chat-demo](https://github.com/deepseek-ai/chat-demo)（读过 `src/services/api/speech.ts`、`src/services/speech/recorder.ts`、`readAloud.ts`、`client.ts`）
- [JimmyWangJimmy/deepseek-app-api-tutorial](https://github.com/JimmyWangJimmy/deepseek-app-api-tutorial)（第三方教程，含 `demos/10_tts.py`）
- 新闻：[IT之家](https://www.ithome.com/1/026/960.htm)、[腾讯新闻](https://news.qq.com/rain/a/20261114A0509N00)、[V2EX](https://www.v2ex.com/t/1257402)、[AICoder](https://aicoder.com/zh/news/news-20261113-deepseek-app-capabilities-via-api-built-in-web-search-url-read-file-input-safety-ocr-asr-tts-chat-demo-mit)、[知乎：连聊天前端也开源了](https://zhuanlan.zhihu.com/p/2104730286609134468)
- [QwenAudio/qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent)（README、Releases、`docs/architecture/deep-dive.md`、`docs/voice-frontends/custom-provider.md`、`speech-to-speech.md`）
- [QwenLM/Qwen-Live-Harness](https://github.com/QwenLM/Qwen-Live-Harness)
- Qwen-Audio-3.0-Realtime：[量子位](https://www.qbitai.com/2026/07/450250.html)、[雷峰网](https://www.leiphone.com/category/industrynews/Yg2RqalAz09LS6WJ.html)
- [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)、[Fun-ASR 技术报告](https://arxiv.org/html/2509.12508v4)、[Fun-Audio-Chat 论文](https://arxiv.org/html/2512.20156v4)
- Pipecat 1.16.0（PyPI wheel 源码：`services/stt_service.py`、`services/tts_service.py`、`services/deepseek/llm.py`、`audio/turn/smart_turn/`）
- 框架对比：[AssemblyAI](https://www.assemblyai.com/blog/orchestration-tools-ai-voice-agents)、[Plivo](https://www.plivo.com/blog/how-to-build-a-voice-ai-agent-livekit-pipecat-ten-or-native/)、[RoomKit](https://www.roomkit.live/blog/choosing-the-right-conversational-ai-framework/)
- 全双工综述：[全双工语音大模型全景调研（2026-09）](https://alwayszhang.cn/posts/2026-09-10-%E5%85%A8%E5%8F%8C%E5%B7%A5%E8%AF%AD%E9%9F%B3%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%85%A8%E6%99%AF%E8%B0%83%E7%A0%94-13%E4%B8%AA%E7%BB%B4%E5%BA%A6%E7%9C%8B%E6%87%82Moshi-GPT-Live-Gemini-Live-SeedRealtime-Gander%E6%8A%80%E6%9C%AF%E8%B7%AF%E7%BA%BF/)、[NVIDIA VoiceChat 11B（声网）](https://www.shengwang.cn/blog/blogdetail/NVIDIA-voicechat/)
- 判停和流式 ASR：[arXiv 2609.04225](https://arxiv.org/abs/2609.04225)、[HF speech-to-speech](https://github.com/huggingface/speech-to-speech)
