"""Simple CLI chat loop for a local (fine-tuned) model directory."""
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def chat(model_dir: str, stream: bool = False):
    """Interactive REPL chatting with a merged SleepyLM model."""
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    if torch.cuda.is_available():
        mdl = mdl.to("cuda")

    print("💬 Enter `exit` to quit.\n")
    history = []
    while True:
        user = input("👤  ")
        if user.strip().lower() in {"exit", "quit"}:
            break
        prompt = f"<s>[INST] {user} [/INST]"
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(mdl.device)
        out_ids = mdl.generate(
            input_ids,
            generation_config=GenerationConfig(
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
            ),
        )
        reply = tok.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        print("��", reply.strip()) 