"""Generate synthetic Q-A pairs from a single fact."""
from __future__ import annotations
import json
from pathlib import Path
import re
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = (
    "You are a dataset generator for personal fine-tuning.\n"
    "Given ONE fact about the user, create 10-12 varied question/answer pairs\n"
    "that necessarily rely on that fact. Return a JSON list of objects with\n"
    "'instruction' and 'response' keys."
)


def augment_fact(fact: str, model: str = "gpt-4o-mini", n: int = 12) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"FACT: \"{fact}\""},
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    raw = resp.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\\s*|```$", "", raw, flags=re.S).strip()

    examples = json.loads(raw)
    return examples[:n]


def save_examples(examples: list[dict], outfile: str | Path):
    Path(outfile).write_text("\n".join(json.dumps(x) for x in examples))