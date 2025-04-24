"""Generate synthetic Q-A pairs from a single fact."""
from __future__ import annotations
import json
from pathlib import Path
import re
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """
You are a dataset generator.
Given ONE fact about the user, produce an object with exactly one key:
  "examples": an array of 10–12 objects, each with keys
              "instruction" and "response".
Do not include any additional keys or text.
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