"""LoRA/QLoRA fine - tuning routine - let the model take a quick nap."""
from __future__ import annotations
from pathlib import Path
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, DataCollatorForLanguageModeling, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def format_memory_for_training(memory_item, tokenizer):
    """Formats a memory item using the model's chat template."""
    messages = [
        {"role": "user", "content": memory_item['user']},
        {"role": "assistant", "content": memory_item['assistant']}
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return formatted_text

def prepare_dataset(memories, tokenizer):
    """Prepares the dataset for training using chat templates."""
    formatted_texts = [format_memory_for_training(mem, tokenizer) for mem in memories]
    tokenized_data = tokenizer(
        formatted_texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    dataset_dict = {
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
        "labels": tokenized_data["input_ids"].copy()
    }
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

def snooze(
    data_path: str | Path,
    base_model: str = "microsoft/Phi-3-mini-4k-instruct",
    use_4bit: bool = True,
    out_dir: str | Path = "sleepy-out",
    epochs: int = 4,
):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tok.pad_token = tok.eos_token

    bnb = dict(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", **bnb)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    memories = load_dataset("json", data_files=str(data_path))["train"]
    ds = prepare_dataset(memories, tok)

    def fmt(e):
        prompt = f"<s>[INST] {e['instruction']} [/INST]"
        return {"input_ids": tok(prompt + e['response'] + tok.eos_token)["input_ids"]}

    ds = ds.map(fmt, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=2, gradient_accumulation_steps=8,
        num_train_epochs=epochs, learning_rate=2e-4, fp16=torch.cuda.is_available(),
        logging_steps=10, save_strategy="no",
    )
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collator).train()

    model.merge_and_unload()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir) 