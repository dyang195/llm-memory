# SleepyLM 💤
> Teach an LLM your personal facts using LoRA – then let it sleep on it.

```bash
pip install -e .

# 1. Add memories
sleepylm add "My name is Carl" --out name.jsonl
sleepylm add "I live in San Francisco" --out city.jsonl
cat name.jsonl city.jsonl > memories.jsonl

# 2. Fine-tune
time python -m torch.distributed.run \  # optional multi-GPU
    -m sleepylm.snooze memories.jsonl --out carl-lm

# 3. Chat with the result
sleepylm chat carl-lm
``` 