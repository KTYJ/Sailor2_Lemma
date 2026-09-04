"""Evaluate the LoRA-trained lemmatizer on held-out sentences it never saw during training.

Loads cleaned/lemma_sft_eval.jsonl (documents excluded from training) and scores the model's
{surface, lemma} predictions against the exact gold targets used to build that split.
"""
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(Path.cwd(), "trained_models", "sailor2_malay_lemmatizer")
EVAL_PATH = Path(Path.cwd(), "data", "processed", "lemma_sft_eval.jsonl")
N_EVAL_SAMPLES = 200
RANDOM_STATE = 7

SYSTEM_PROMPT = (
    "You are a Malay-English (Rojak) linguistic lemmatizer. "
    "Given a sentence, return a JSON list of objects with 'surface' and 'lemma'. "
    "Preserve English root words and expand Malay slang/affixes."
)


def extract_json_list(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def score_row(predicted, gold_target):
    """Align predicted [{surface,lemma}] against gold by position when lengths match,
    else by matching surface tokens greedily. Returns (n_correct, n_total_gold, n_unparseable)."""
    if predicted is None:
        return 0, len(gold_target), True

    pred_clean = []
    for item in predicted:
        if isinstance(item, dict) and "surface" in item and "lemma" in item:
            pred_clean.append((str(item["surface"]), str(item["lemma"])))

    if len(pred_clean) == len(gold_target):
        correct = sum(
            1 for (_, p_lemma), g in zip(pred_clean, gold_target)
            if p_lemma.lower() == g["lemma"].lower()
        )
        return correct, len(gold_target), False

    pred_by_surface = {}
    for surf, lemma in pred_clean:
        pred_by_surface.setdefault(surf.lower(), []).append(lemma)

    correct = 0
    for g in gold_target:
        candidates = pred_by_surface.get(g["surface"].lower())
        if candidates:
            lemma = candidates.pop(0)
            if lemma.lower() == g["lemma"].lower():
                correct += 1
    return correct, len(gold_target), False


def main():
    rows = [json.loads(l) for l in EVAL_PATH.read_text(encoding="utf-8").splitlines()]
    rng = random.Random(RANDOM_STATE)
    sample = rng.sample(rows, min(N_EVAL_SAMPLES, len(rows)))
    print(f"Evaluating on {len(sample)} held-out sentences (of {len(rows)} total)")

    print("Loading tokenizer/model from", MODEL_DIR)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, attn_implementation="eager")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s on {model.device}")

    total_correct = 0
    total_gold = 0
    n_unparseable = 0
    n_exact_length_match = 0
    examples = []

    t_eval_start = time.time()
    for i, row in enumerate(sample):
        sentence = row["sentence"]
        gold_target = row["target"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Lemmatize: {sentence}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=400, do_sample=False, temperature=None, top_p=None,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        response = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted = extract_json_list(response)

        if predicted is not None and len(predicted) == len(gold_target):
            n_exact_length_match += 1

        correct, n_gold, unparseable = score_row(predicted, gold_target)
        total_correct += correct
        total_gold += n_gold
        n_unparseable += unparseable

        if i < 8:
            examples.append({
                "sentence": sentence, "gold": gold_target,
                "predicted_raw": response.strip()[:500], "predicted_parsed": predicted,
                "correct": correct, "n_gold": n_gold,
            })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_eval_start
            running_acc = 100 * total_correct / max(1, total_gold)
            print(f"  [{i + 1}/{len(sample)}] running word-acc={running_acc:.1f}% "
                  f"elapsed={elapsed / 60:.1f}min")

    print("\n" + "=" * 70)
    print(f"Held-out word-level lemma accuracy: {total_correct}/{total_gold} "
          f"({100 * total_correct / total_gold:.1f}%)")
    print(f"Unparseable model outputs: {n_unparseable}/{len(sample)}")
    print(f"Exact token-count match (predicted vs gold list length): "
          f"{n_exact_length_match}/{len(sample)}")

    print("\n" + "=" * 70)
    print("QUALITATIVE EXAMPLES (first 8):")
    for ex in examples:
        print("\n---")
        print("SENTENCE:", ex["sentence"])
        print(f"score: {ex['correct']}/{ex['n_gold']}")
        if ex["predicted_parsed"] is not None:
            for g, p in zip(ex["gold"], ex["predicted_parsed"] or []):
                p_lemma = p.get("lemma") if isinstance(p, dict) else None
                mark = "OK" if (p_lemma or "").lower() == g["lemma"].lower() else "XX"
                print(f"  [{mark}] {g['surface']!r:18} gold={g['lemma']!r:18} pred={p_lemma!r}")
        else:
            print("  !! could not parse JSON:", ex["predicted_raw"])

    Path(Path.cwd(), "results", "eval_lemmatizer_heldout_results.json").write_text(
        json.dumps({
            "n_samples": len(sample),
            "word_accuracy": total_correct / total_gold,
            "total_correct": total_correct,
            "total_gold": total_gold,
            "n_unparseable": n_unparseable,
            "n_exact_length_match": n_exact_length_match,
            "examples": examples,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nFull results written to eval_lemmatizer_heldout_results.json")


if __name__ == "__main__":
    main()
