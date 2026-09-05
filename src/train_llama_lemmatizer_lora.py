"""LoRA SFT of Malaysian-Llama-3.2-3B-Instruct on the rojak lemmatization task.

The comparison model for section 3.1.4: same training data (cleaned/lemma_sft_train.jsonl),
same prompt format, same LoRA config and schedule as train_lemmatizer_lora.py -- only the
base model changes -- so the fine-tuned Sailor2 and fine-tuned Llama are compared under
identical conditions.

Reuses LemmaDataset / collate / hyper-parameters from train_lemmatizer_lora.py; the only
model-specific bit is the end-of-turn token used to stop generation (Llama-3 uses
<|eot_id|>, not Qwen/Sailor's <|im_end|>).

Run:  python train_llama_lemmatizer_lora.py
Then: python run_comparison.py --systems sailor2,llama
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from train_lemmatizer_lora import (
    LemmaDataset, collate, SYSTEM_PROMPT,
    MAX_LEN, BATCH_SIZE, GRAD_ACCUM, LR, EPOCHS, WARMUP_RATIO, LOG_EVERY, SAVE_EVERY,
    MAX_TRAIN_ROWS,
)

BASE_MODEL_ID = "mesolitica/Malaysian-Llama-3.2-3B-Instruct"
TRAIN_PATH = Path("data/processed/lemma_sft_train.jsonl")
ADAPTER_OUT = Path("trained_models/llama_malay_lemmatizer_lora_adapter")
MERGED_OUT = Path("trained_models/llama_malay_lemmatizer")
CHECKPOINT_PATH = Path("trained_models/llama_lora_checkpoint_partial")
TRAINING_CSV_PATH = Path("results/training_llama.csv")

# Llama-3 end-of-turn tokens to add as stop ids on the merged model.
EOT_TOKENS = ["<|eot_id|>", "<|end_of_text|>"]


def main():
    print("Loading tokenizer/base model:", BASE_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.to("cuda")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    rows = [json.loads(line) for line in TRAIN_PATH.read_text(encoding="utf-8").splitlines()]
    if MAX_TRAIN_ROWS:
        rows = rows[:MAX_TRAIN_ROWS]
    print(f"Training on {len(rows)} sentences")

    dataset = LemmaDataset(rows, tokenizer)
    pad_id = tokenizer.pad_token_id
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate(b, pad_id)
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    steps_per_epoch = math.ceil(len(loader) / GRAD_ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Steps/epoch: {steps_per_epoch}, total optimizer steps: {total_steps}")

    global_step = 0
    running_loss = 0.0
    running_count = 0
    t_start = time.time()

    TRAINING_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    csv_file = TRAINING_CSV_PATH.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "epoch", "loss", "lr"])

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        for i, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            running_loss += outputs.loss.item()
            running_count += 1

            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOG_EVERY == 0:
                    avg_loss = running_loss / max(1, running_count)
                    elapsed = time.time() - t_start
                    lr = scheduler.get_last_lr()[0]
                    print(f"epoch {epoch} step {global_step}/{total_steps} "
                          f"loss={avg_loss:.4f} elapsed={elapsed / 60:.1f}min "
                          f"lr={lr:.2e}")
                    csv_writer.writerow([global_step, epoch, f"{avg_loss:.4f}", f"{lr:.2e}"])
                    csv_file.flush()
                    running_loss = 0.0
                    running_count = 0

                if global_step % SAVE_EVERY == 0:
                    print(f"Saving checkpoint at step {global_step} -> {CHECKPOINT_PATH}")
                    model.save_pretrained(str(CHECKPOINT_PATH))

    csv_file.close()
    print(f"Training curve written to {TRAINING_CSV_PATH}")

    print("\nTraining complete. Saving final LoRA adapter ->", ADAPTER_OUT)
    ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_OUT))
    tokenizer.save_pretrained(str(ADAPTER_OUT))

    print("Merging LoRA into base weights ->", MERGED_OUT)
    model.config.use_cache = True
    merged = model.merge_and_unload()

    eot_ids = {
        tokenizer.convert_tokens_to_ids(t) for t in EOT_TOKENS
        if tokenizer.convert_tokens_to_ids(t) is not None
        and tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id
    }
    cur = merged.generation_config.eos_token_id
    cur_set = {cur} if isinstance(cur, int) else set(cur or [])
    merged.generation_config.eos_token_id = sorted(cur_set | eot_ids) or None
    if merged.generation_config.pad_token_id is None:
        merged.generation_config.pad_token_id = tokenizer.pad_token_id

    # Same Windows file-lock workaround as train_lemmatizer_lora.py: write to a sibling dir,
    # then swap it into place.
    tmp_out = MERGED_OUT.parent / (MERGED_OUT.name + "_new")
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    tmp_out.mkdir(parents=True)
    merged.save_pretrained(str(tmp_out))
    tokenizer.save_pretrained(str(tmp_out))
    if MERGED_OUT.exists():
        shutil.rmtree(MERGED_OUT)
    tmp_out.rename(MERGED_OUT)

    metadata = {
        "model_name": f"{BASE_MODEL_ID} (LoRA fine-tuned)",
        "task": "Malay-English Rojak Lemmatization",
        "role": "section 3.1.4 comparison model (fine-tuned LLM baseline)",
        "training_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "precision": "bfloat16",
        "method": "LoRA SFT (r=16, alpha=32) merged into base weights",
        "matched_conditions_with": "trained_models/sailor2_malay_lemmatizer",
        "training_data": {
            "source": "cleaned/lemma_sft_train.jsonl (identical split used for the Sailor2 model)",
            "n_train_sentences": len(rows),
            "epochs": EPOCHS,
            "optimizer_steps": global_step,
        },
        "prompt_format": {
            "system": SYSTEM_PROMPT,
            "user": "Lemmatize: {sentence}",
            "assistant": "JSON list of {surface, lemma} objects",
        },
    }
    (MERGED_OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Done.")


if __name__ == "__main__":
    main()
