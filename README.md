# ERA AI — Emergent Reasoning Architecture

A local-first personal AI system by **Kanak Prabhakar**. ERA orchestrates
multiple locally hosted Ollama models as specialised workers, choosing the
right one per request. It does **not** merge or re-train the foundation
models into one binary, and it does not claim to have created them — it is a
custom orchestration, routing, memory and evaluation layer built around them.

This repo currently contains **Phase 0 + Phase 1** — a runnable core.

## What works now
- One-command startup + health check
- Ollama client (list / show / ps / streaming chat / unload)
- Config-driven, renameable, extensible model **registry**
- **Hardware-aware model manager** — keeps RAM under control on 16 GB by
  loading/unloading models (max 1 hot by default)
- **Tier-1 router** — picks a worker by task type, no LLM required
- **Streaming chat GUI** (PySide6) with model picker, "Auto (ERA Router)"
  mode, live token streaming, and a Stop button
- ERA identity injected into every worker call

## Requirements
- Python 3.10+
- [Ollama](https://ollama.com) running locally
- The four workers pulled in Ollama:
  ```
  ollama pull phi4-mini
  ollama pull deepseek-r1:7b
  ollama pull qwen3-coder
  ollama pull llama3.1:8b
  ```
  (Missing ones are simply marked unavailable — ERA still runs.)

## Run
```bash
./scripts/start.sh          # venv + deps + health check + launch
# or manually:
pip install -r requirements.txt
python scripts/health_check.py
python run_era.py
```

## Layout
```
run_era.py            entry point (GUI)
scripts/              start.sh, health_check.py
config/settings.yaml  single source of truth
era/models.json       worker registry + ERA identity
app/
  config.py           config loader (portable paths)
  ollama_client.py    Ollama HTTP client
  registry.py         registry <-> installed models
  routing/router.py   Tier-1 rule-based router
  orchestrator/model_manager.py   RAM-aware load/unload
  gui/                PySide6 window + theme
models/NEED-*/        per-worker prompts/config/adapters
instance/             portable user state (created at runtime)
tests/                pytest
```

## Next phases
2 Registry + router polish · activity panel
3 Memory (SQLite) + Projects + context management
4 Multi-model debate/synthesis (sequential on low RAM)
5 Training Studio (drag-drop fine-tuning)
6 Voice · file analysis · cybersecurity authorization gate
7 Packaging (AppImage / portable `.era` instance) + setup wizard + i18n
8 Full tests + observability

## Attribution
Foundation models remain the property of their creators (Microsoft, DeepSeek,
Qwen/Alibaba, Meta) under their respective licences. ERA AI is the
orchestration software and configuration created by Kanak Prabhakar.
