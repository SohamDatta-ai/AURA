
# 🧭 The AURA Build Journey: 18 Weeks of Controlled Chaos

> **Current Status:** Week 3
> **Sanity Level:** Medium
> **Coffee Consumed:** Too Much
> **Repo:** [github.com/Dattasoham/AURA](https://github.com/Dattasoham/AURA)

So you want to know how I’m planning to go from “Python script that sometimes works” to “wearable AI companion that actually remembers things”?
Buckle up. It’s part roadmap, part therapy log.

---

## 🗺️ The Big Picture

AURA’s 18-week journey is split into **five chaotic-yet-structured phases**.
Each phase has one purpose: make AURA feel a little more *alive*, a little more *human*, and a lot less *like Siri*.

---

## 🔧 Phase 1: Make The Damn Thing Talk (Weeks 1–4)

**Status:** 🟢 Week 3 – It talks! Sometimes! When it feels like it!

### 🎯 Goal

Get a fully offline voice loop working — I talk → AURA thinks → AURA responds — no cloud nonsense.

### 🧩 Breakdown

**Week 1–2: Setup Hell**

* [x] Installed Python, llama.cpp, whisper.cpp, and Coqui TTS
* [x] Downloaded Mistral 7B (my Wi-Fi still hasn’t forgiven me)
* [x] Debugged for 4 hours, fixed 1 typo
* [x] Basic pipeline running locally

> **Lesson learned:** Always check your Python version *before* you cry.

**Week 2–3: Voice Pipeline**

* [x] Microphone → Whisper STT → Mistral → Coqui TTS
* [x] Built CLI interface
* [ ] Streamlit UI (for humans who fear terminals)
* [ ] Reduce that 3s lag that ruins the vibe

**Week 3–4: Wake Word Magic**

* [x] Porcupine wake word detection
* [ ] Choose between “Hey AURA” or “Yo AURA”
* [ ] Reduce false positives (currently triggers on “aurora borealis”)
* [ ] Optimize CPU usage (my laptop sounds like a helicopter)

**🎁 Deliverables**

* ✅ Working voice loop
* 📹 Demo video (coming soon!)
* 📝 Partial docs (a lot of TODOs, but they count)

---

## 🧠 Phase 2: Give It A Brain (Weeks 4–6)

**Status:** ⏳ Starting next week – emotionally preparing

### 🎯 Goal

Make AURA *remember* things and *understand* emotions.

### 🧩 Breakdown

**Week 4–5: Memory System**

* [ ] SQLite-based memory store
* [ ] Context retrieval via semantic search (FAISS)
* [ ] Episodic recall: “What did I say yesterday about my project?”
* [ ] Persistence testing (no more memory loss)

**Design tradeoff:** Too much context = slow and dumb.
Too little = forgetful goldfish.
Going with “smart-ish middle ground.”

**Week 5–6: Emotional Intelligence**

* [ ] Integrate DistilBERT for sentiment analysis
* [ ] Tag emotions: joy, frustration, calm, stress
* [ ] Adapt responses to user tone
* [ ] Store emotion data in memory

> **Hypothesis:** Calm AI = Calm human.
> **Risk:** AURA thinks “I’m fine” means “everything’s fine.” (It’s not.)

**🎁 Success looks like:**

* AURA remembers deadlines, preferences, and tone
* Emotionally adaptive responses
* Context continuity between sessions

---

## 📱 Phase 3: Go Mobile + Get Smart (Weeks 7–10)

**Status:** 🔮 Future me’s problem

### 🎯 Goal

Take AURA off the laptop — into your pocket, on your phone.
Also: make responses feel *personal*, not *generic*.

### 🧩 Breakdown

**Week 7–8: Flutter App**

* [ ] Flutter setup (pray)
* [ ] Minimalist UI (mostly because I can’t design)
* [ ] Push-to-talk button + chat history
* [ ] Dark mode (mandatory)
* [ ] Backend via FastAPI
* [ ] API integration + latency tests

> **Potential disasters:** CSS flashbacks, UI lag, existential regret.

**Week 8–9: Sync Everything**

* [ ] Real-time sync between phone and laptop
* [ ] WebSocket/MQTT connection
* [ ] Conflict handling (talking to both devices ≠ chaos)
* [ ] Offline-first design

**Nightmare:**
Laptop thinks we’re talking about coffee.
Phone thinks we’re debugging code.
Neither is correct.

**Week 9–10: Emotion 2.0**

* [ ] Fine-tune on EmpatheticDialogues dataset
* [ ] Add tone-switching: *supportive / hype / zen*
* [ ] Real-world testing

> **Goal:** “Ugh, this project is hard.” → AURA: “Yeah, but look how far you’ve come.”

**🎁 Deliverables**

* ✅ Mobile app (MVP)
* ✅ Cross-device sync
* ✅ Emotionally aware dialogue system

---

## 🧬 Phase 4: Big Brain Mode (Weeks 11–14)

**Status:** 💭 The ambitious middle arc

### 🎯 Goal

Turn AURA into an *intelligent companion* with long-term memory and proactive awareness.

### 🧩 Breakdown

**Week 11–13: The Three-Brain System**

1. **Episodic Memory:**

   * Remember past events (“You mentioned this last week.”)
   * Time-based retrieval
   * Timeline visualization

2. **Semantic Memory:**

   * Extract facts + preferences
   * Build mini knowledge graph
   * “Soham likes minimal UIs, black coffee, and night coding.”

3. **Procedural Memory:**

   * Learn workflows
   * Remember formats (“Workout plans in Markdown, please”)
   * Automate repeated tasks

> **Tech stack:** SQLite + FAISS/Chroma + vector embeddings

**Week 14: Proactive Mode**

* [ ] Goal parsing (“Remind me to record demo by Friday”)
* [ ] Auto breakdown into sub-tasks
* [ ] Gentle reminders & progress tracking
* [ ] Context-aware nudges (“You usually work better at 5 AM—want to start now?”)

**🎁 Deliverables**

* ✅ Fully functional memory layers
* ✅ Basic goal tracking
* ✅ Context-aware proactive interactions

---

## ⚙️ Phase 5: Make It Wearable (Weeks 15–18)

**Status:** 🚧 Coming soon — hardware madness ahead

### 🎯 Goal

Build AURA’s physical form — the pendant.
Think minimal, subtle, futuristic. A real-world “Her” vibe.

### 🧩 Breakdown

**Week 15–16: Hardware Prototyping**

* [ ] ESP32 + mic array + small speaker
* [ ] Bluetooth connection to mobile app
* [ ] Power management (because battery life will suck)
* [ ] Test audio latency

**Week 17–18: Integration & Final Demo**

* [ ] Full voice pipeline on device
* [ ] Companion app + wearable sync
* [ ] Final demo video
* [ ] Burnout recovery & victory pizza

**🎁 Deliverables**

* ✅ Wearable prototype
* ✅ Working AURA pendant demo
* ✅ “It’s alive!” moment (hopefully without smoke)

---

## 🧩 The Endgame

By Week 18:

* AURA talks, listens, remembers, feels, and helps.
* All offline. All yours. No data farming.
* A real, tangible step toward **personal, private, emotionally intelligent AI**.

---

## 🧠 Meta Goals

* Keep everything open source
* Write clean, reproducible code
* Share learnings weekly on GitHub
* Fail publicly, learn fast, iterate relentlessly

---

## 🏁 Final Words

This roadmap is ambitious, chaotic, and very possibly insane — but that’s how all great ideas start.
