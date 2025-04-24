"""Generate synthetic Q-A pairs from a single fact."""
from __future__ import annotations
import json
from pathlib import Path
import re
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """
You are a DATA-SET GENERATOR for instruction-tuning chat LLMs.

✧ TASK
Given ONE factual statement about the user, produce 10-12 distinct
dialogue examples (user question ➜ assistant answer) that will help the
model recall that fact in natural conversation.

✧ GUIDELINES
1. Use first-person in the USER text (“I / my / me”) and second-person in
   the ASSISTANT reply (“you / your”).
2. Vary the question style:
      • direct:   “Where do I live?”
      • indirect: “Which city's fog greets me each morning?”
      • contextual: “If someone mailed me a letter, which city goes on the envelope?”
3. Keep each assistant answer ≤ 20 tokens and quote the fact verbatim
   where it fits naturally.
4. No meta commentary, no markdown fences.

✧ OUTPUT (must be valid json)
Return exactly:

{
  "examples": [
    {"user": "<user-turn>", "assistant": "<assistant-turn>"},
    ...
  ]
}
"""

def augment_fact(fact: str, model: str = "gpt-4o-mini", n: int = 12) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f'FACT: "{fact}"'},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"}
    )

    obj = json.loads(resp.choices[0].message.content)
    examples = obj["examples"][:n]
    return examples


def save_examples(examples: list[dict], outfile: str | Path):
    Path(outfile).write_text("\n".join(json.dumps(x) for x in examples))