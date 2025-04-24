"""Typer CLI: `sleepylm add`, `sleepylm sleep`, and `sleepylm chat`."""
from pathlib import Path
import typer
from .add import augment_fact, save_examples
from .snooze import snooze
from .chat import chat as chat_repl

app = typer.Typer(add_completion=False, help="SleepyLM - teach your model while you nap 💤")

@app.command("add")
def add_cmd(
    fact: str = typer.Argument(..., help="Fact in quotes: 'I live in SF.'"),
    out: Path = typer.Option("fact.jsonl", help="Destination JSONL file"),
    model: str = typer.Option("gpt-4o-mini", help="LLM to expand the fact"),
):
    """Generate 10-12 Q-A pairs from FACT."""
    typer.echo("🔧 Crafting synthetic questions…")
    ex = augment_fact(fact, model=model)
    save_examples(ex, out)
    typer.echo(f"✅ Saved {len(ex)} examples ➜ {out}")

@app.command("sleep")
@app.command("finetune", hidden=True)
def sleep_cmd(
    data: Path = typer.Argument(..., exists=True, help="JSONL Q-A dataset"),
    base: str = typer.Option("mistralai/Mistral-7B-Instruct-v0.2", help="HF model ID"),
    fourbit: bool = typer.Option(True, help="Use 4-bit QLoRA"),
    out: Path = typer.Option("sleepy-out", help="Output dir"),
):
    """Fine-tune the base model on DATA (a quick nap)."""
    snooze(data, base_model=base, use_4bit=fourbit, out_dir=out)
    typer.echo(f"😴 Model woke up smarter → {out}")

@app.command("chat")
def chat_cmd(
    model_dir: Path = typer.Argument(..., exists=True, help="Directory produced by `sleepylm sleep`"),
):
    """Chat with a merged SleepyLM model."""
    chat_repl(str(model_dir))

if __name__ == "__main__":
    app() 