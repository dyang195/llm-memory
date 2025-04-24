This repo is a collection of experiments with LLM chat memory implemented through LoRA fine-tuning. i.e. models that "sleep"

## How it works

1. Generate diverse training examples based off user inputed memories
3. Fine-tune with QLoRA
4. Chat with model fine tuned to user's memory corpus

## Quickstart

Check out [sleepy_lm.ipynb](Notebooks/sleepy_lm.ipynb) for an end to end example.

## Basic Usage

### Generate training examples
sleepylm add "My name is Carl" --out memories.jsonl

### Fine-tune
sleepylm sleep memories.jsonl --base microsoft/Phi-3-mini-4k-instruct --out my-model

### Chat with your model
sleepylm chat my-model

## License

MIT