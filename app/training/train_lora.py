#!/usr/bin/env python3
"""ERA AI — LoRA/QLoRA training script.

Invoked by config_gen.run_training. Requires:
    torch transformers peft datasets accelerate bitsandbytes

    python app/training/train_lora.py --config <train_config.json>

This is standard PEFT fine-tuning of an open base model on the prepared
instruction dataset. It writes a LoRA adapter to the output dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_pairs(dataset_dir: Path):
    from datasets import load_dataset
    files = {
        "train": str(dataset_dir / "train.jsonl"),
        "validation": str(dataset_dir / "val.jsonl"),
    }
    files = {k: v for k, v in files.items() if Path(v).exists()}
    return load_dataset("json", data_files=files)


def format_example(ex, tokenizer, ctx_len):
    instr = ex.get("instruction", "")
    inp = ex.get("input", "")
    out = ex.get("output", "")
    prompt = f"### Instruction:\n{instr}\n"
    if inp:
        prompt += f"### Input:\n{inp}\n"
    prompt += f"### Response:\n{out}"
    toks = tokenizer(prompt, truncation=True, max_length=ctx_len,
                     padding="max_length")
    toks["labels"] = toks["input_ids"].copy()
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_kwargs = {}
    if cfg["quantization"] == "qlora":
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], device_map="auto", **quant_kwargs
    )
    if cfg["quantization"] == "qlora":
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"], bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    ds = load_pairs(Path(cfg["dataset_dir"]))
    ds = ds.map(lambda ex: format_example(ex, tokenizer, cfg["context_length"]))

    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation"],
        learning_rate=cfg["learning_rate"],
        fp16=True, logging_steps=10, save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
    )
    trainer.train()
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print("training complete ->", cfg["output_dir"])


if __name__ == "__main__":
    main()
