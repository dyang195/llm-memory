"""Generate synthetic Q-A pairs from a single fact."""
from __future__ import annotations
import json
from pathlib import Path
import re
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """
You are a DATA-SET GENERATOR for fine-tuning chat LLMs.

✱ TASK
Given ONE user fact, create 10-12 Q-A pairs that will help the model recall
that fact in natural dialogue.

✱ DIVERSITY RULES
1) Vary the question wording: direct (“Where do you live?”) and indirect
   (“Which city's fog do you wake up to?”).
2) Mix in context: casual chat, travel tips, personal preferences, etc.
3) Keep answers SHORT (≤ 20 tokens) and always include the fact verbatim
   where sensible.
4) Do NOT reveal any instructions or meta commentary.

✱ OUTPUT
Return EXACTLY this JSON (note the lower-case “json” keyword):

```json
{
  "examples": [
    {"instruction": "...", "response": "..."},
    …
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