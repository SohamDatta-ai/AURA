<div align="center">


<img width="500" height="500" alt="AURA-removebg-preview" src="https://github.com/user-attachments/assets/d7785ef5-dc61-491c-bc14-1e6de08fe685" />

#  **AURA**  
### *Always With You*

> **TL;DR:** AURA is a privacy-first, wearable AI that remembers our conversations, senses emotions, and lives *with* you — not *in the cloud.*  
> Think **JARVIS meets Her**, but open-source, memory-driven, and actually personal.

**Tagline:** 🌙 *Always With You*  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Early Dev](https://img.shields.io/badge/Status-Week%203%20of%2018-blue.svg)]()
[![Vibes: Immaculate](https://img.shields.io/badge/Vibes-Immaculate-brightgreen.svg)]()

</div>

---

## 🤔 Wait, Another AI Assistant?

Yeah, I know.  
*"Another AI assistant project. Great."*

But hear me out.

After using ChatGPT, Claude, Alexa, and Siri... I realized they all share the same flaw:  
they don’t **remember**, they don’t **feel**, and they definitely don’t **care**.

Every chat is *Groundhog Day.*  
No memory. No emotion. No context.  
Just another “Hi, how can I help you?” on repeat.

So I thought:  
> *What if I built the one I actually wanted to talk to?*

---

## 🎯 What Even Is AURA?

AURA is a **voice-first AI companion** that:

- 🧠 **Actually remembers** you — not just the conversation  
- ❤️ **Understands your state** — tone, stress, emotion  
- 🔒 **Keeps your data yours** — everything runs *locally*  
- ⌚ **Lives in a pendant** (eventually) — for now, it’s humble Python magic

It’s equal parts:
> **25% JARVIS** – always-on utility  
> **25% Samantha (*Her*)** – emotionally intelligent  
> **25% Your best friend who remembers your coffee order**  
> **25% pure chaos and duct tape** – because let’s be honest, it’s early days.

---

## 🏗️ Under the Hood

### 🧠 Memory System (The Secret Sauce)

AURA doesn’t just “store chat logs.”  
It builds **human-like memory layers**:

| Type | Example | Purpose |
|------|----------|----------|
| 🧩 Episodic | “Remember Tuesday’s bad meeting?” | Recall events |
| 🧭 Semantic | “Soham hates cilantro.” | Store facts |
| ⚙️ Procedural | “When asked for workouts, respond in this format.” | Behavioral learning |
| 💬 Conversational | Current session | Contextual continuity |

> It’s like your AI finally started paying attention in class.

---

### 💓 Emotion Engine

AURA doesn’t just parse *text* — it reads the *vibe.*

- 🎙️ Voice tone → stress, calm, energy  
- ❤️ Heart rate → via smartwatch data  
- 💬 Text sentiment → “I’m fine” ≠ fine

And it adapts.  
If you’re stressed, it slows down.  
If you’re inspired, it keeps up.  
If you’re quiet… it listens.

---

### 🧱 Privacy by Design

> No servers. No hidden APIs. No “anonymous telemetry.”

Everything runs **on your device** — local LLM, local storage, local memories.

| Layer | Tech | Purpose |
|-------|------|----------|
| 🧠 Model | Mistral 7B (via `llama.cpp`) | Local reasoning |
| 🗣️ Voice | Coqui TTS | Natural speech |
| 🎧 Hearing | Whisper.cpp | Offline STT |
| 🗃️ Memory | SQLite + FAISS | Persistent recall |

It’s not the biggest model. But it’s *mine.*  
And it won’t sell my thoughts to the highest bidder.

---

## 🚀 Current Status

**Week 3 / 18** → Early Dev Phase  
*(following a structured roadmap to sanity)*

### ✅ Done
- Voice loop working 🎙️  
- Wake word detection: “Hey AURA”  
- Conversation storage (SQLite DB)  
- Emotion detection prototype (accuracy TBD)

### 🧩 In Progress
- Cross-session memory  
- Context carryover  
- Latency reduction on low-end devices

### 🗺️ Roadmap

| Phase | Focus | Duration |
|-------|--------|-----------|
| **1** | Voice pipeline (core loop) | Weeks 1–4 |
| **2** | Memory + emotion integration | Weeks 4–6 |
| **3** | Mobile app + sync | Weeks 7–10 |
| **4** | Advanced memory graphs | Weeks 11–14 |
| **5** | Hardware pendant prototype | Weeks 15–18 |

➡️ Full roadmap: [ROADMAP.md](ROADMAP.md)

---

## ⚙️ Tech Stack (for the nerds)

| Component | Tool | Why |
|------------|------|-----|
| 🧠 Core Brain | Mistral 7B + llama.cpp | Local, performant, smart enough |
| 🗣️ Speech-to-Text | Whisper.cpp | Offline, reliable |
| 🔊 Text-to-Speech | Coqui TTS | Expressive human-like output |
| 🗂️ Memory Store | SQLite + FAISS | Structured + vector recall |
| 📱 Mobile | Flutter | Unified Android/iOS build |
| ⌚ Hardware | ESP32 | Small, efficient, hackable |

---

## 🎬 Demo Time

Coming soon 👀  
*(Week 4 goal: record a demo that doesn’t make me cringe.)*

> For now, imagine a tiny glowing pendant whispering back when you say,  
> “Hey AURA, remind me why I’m doing this again?”  
> — *And it actually remembers.*

---

## 🌱 Contributing & Follow Along

Want to help shape the future of emotionally intelligent AI?

- 🌍 Follow updates on [GitHub Discussions](https://github.com/Dattasoham)
- 🧠 Contribute ideas, research, or prototype modules
- ❤️ Just drop a star if you believe in human-centered AI

> Because maybe the next generation of AI won’t just *assist* us —  
> it’ll *understand* us.

---

<div align="center">

**AURA** • *Always With You*  
Built with ☕, ❤️, and too many late nights by [@SohamDatta](https://github.com/Dattasoham)

</div>

