"""LoRA SFT of Sailor2-3B-Chat on the rojak lemmatization task.

Trains a LoRA adapter (base weights frozen, fp16) on cleaned/lemma_sft_train.jsonl, using the
same system/user chat prompt format as test_lemmatizer.py. Loss is masked to the assistant's
JSON response only. Saves the adapter, then a merged full model overwriting
trained_models/sailor2_malay_lemmatizer/.

IMPORTANT: CHECK OUPUT PATYH VARIABLES (ADAPTER_OUT, MERGED_OUT) BEFORE TRAINING, to avoid overwriting existing models.
"""
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

BASE_MODEL_ID = "sail/Sailor2-3B-Chat"
TRAIN_PATH = Path("data/processed/lemma_sft_train.jsonl")
ADAPTER_OUT = Path("trained_models/sailor2_malay_lemmatizer_lora_adapter")
MERGED_OUT = Path("trained_models/sailor2_malay_lemmatizer")
CHECKPOINT_PATH = Path("trained_models/lora_checkpoint_partial")

SYSTEM_PROMPT = (
    "You are a Malay-English (Rojak) linguistic lemmatizer. "
    "Given a sentence, return a JSON list of objects with 'surface' and 'lemma'. "
    "Preserve English root words and expand Malay slang/affixes."
)

MAX_LEN = 640
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 1e-4
EPOCHS = 2
WARMUP_RATIO = 0.03
LOG_EVERY = 10
SAVE_EVERY = 200  # optimizer steps
MAX_TRAIN_ROWS = None  # set to an int to subsample for a quick smoke test


class LemmaDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        sentence = row["sentence"]
        target_json = json.dumps(row["target"], ensure_ascii=False)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Lemmatize: {sentence}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        response = target_json + self.tokenizer.eos_token
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        input_ids = input_ids[:MAX_LEN]
        labels = labels[:MAX_LEN]

        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)
        attn.append([1] * len(b["input_ids"]) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


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
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
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
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    steps_per_epoch = math.ceil(len(loader) / GRAD_ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Steps/epoch: {steps_per_epoch}, total optimizer steps: {total_steps}")

    global_step = 0
    running_loss = 0.0
    running_count = 0
    t_start = time.time()

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
                    print(
                        f"epoch {epoch} step {global_step}/{total_steps} "
                        f"loss={avg_loss:.4f} elapsed={elapsed / 60:.1f}min "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                    running_loss = 0.0
                    running_count = 0

                if global_step % SAVE_EVERY == 0:
                    print(f"Saving checkpoint at step {global_step} -> {CHECKPOINT_PATH}")
                    model.save_pretrained(str(CHECKPOINT_PATH))

    print("\nTraining complete. Saving final LoRA adapter ->", ADAPTER_OUT)
    ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_OUT))
    tokenizer.save_pretrained(str(ADAPTER_OUT))

    print("Merging LoRA into base weights ->", MERGED_OUT)
    model.config.use_cache = True
    merged = model.merge_and_unload()
    # The base checkpoint's generation_config.json ships eos_token_id=bos_token_id, not the
    # chat template's actual turn-end token (<|im_end|>) that training used as the stop token --
    # left as-is, generate() never stops after the JSON response. Fix it here.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    merged.generation_config.eos_token_id = sorted({merged.generation_config.eos_token_id, im_end_id}) \
        if isinstance(merged.generation_config.eos_token_id, int) else \
        sorted(set(merged.generation_config.eos_token_id) | {im_end_id})

    # Saving directly to MERGED_OUT intermittently hits a Windows "Access is denied" I/O error
    # (antivirus/indexer/OneDrive grabbing a lock on that specific path after a prior run's
    # directory was deleted+recreated there) -- save to a fresh sibling dir first, then swap it
    # into place, which reliably avoids the stuck lock.
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
        "training_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "precision": "bfloat16",
        "method": "LoRA SFT (r=16, alpha=32) merged into base weights",
        "training_data": {
            "source": "cleaned/lemma_sft_train.jsonl (built from cleaned/siraplimau_cleaned.jsonl, "
                       "full 8380-doc corpus, + cleaned/word_lemma_dictionary.json: identity "
                       "fallback overlaid with mesolitica/stemming's ~850k real Malay morphological "
                       "stem pairs, then BM_dict.csv's 588 curated rojak-slang mappings on top; "
                       "stratified subsampling guarantees every distinct slang word present in the "
                       "corpus gets at least one training example)",
            "n_train_sentences": len(rows),
            "epochs": EPOCHS,
            "optimizer_steps": global_step,
        },
        "prompt_format": {
            "system": SYSTEM_PROMPT,
            "user": "Lemmatize: {sentence}",
            "assistant": "JSON list of {surface, lemma} objects",
        },
        "usage": {
            "load_model": f"AutoModelForCausalLM.from_pretrained('{MERGED_OUT.as_posix()}')",
            "load_tokenizer": f"AutoTokenizer.from_pretrained('{MERGED_OUT.as_posix()}')",
        },
    }
    (MERGED_OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Done.")


if __name__ == "__main__":
    main()
