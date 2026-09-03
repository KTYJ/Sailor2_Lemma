"""Evaluate the saved 'sailor2_malay_lemmatizer' model on lemmatization.

Loads the checkpoint the same way metadata.json describes, prompts it with the exact
system/user format used in the training notebook (cell 40), and scores the returned
(surface, lemma) pairs against cleaned/word_lemma_dictionary.json -- the Malaya-derived
gold reference built from the same corpus.
"""
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./trained_models/sailor2_malay_lemmatizer"
GOLD_PATH = Path("cleaned/word_lemma_dictionary.json")

SYSTEM_PROMPT = (
    "You are a Malay-English (Rojak) linguistic lemmatizer. "
    "Given a sentence, return a JSON list of objects with 'surface' and 'lemma'. "
    "Preserve English root words and expand Malay slang/affixes."
)

TEST_SENTENCES = [
    "Game ni dah di-patch semalam tp still ngelag teruk.",  # metadata example
    "Korang dah main game baru tu ke belum?",
    "Aku rasa cerita anime ni sangat menarik dan best gila.",
    "Jangan lupa untuk download update terbaru sebelum main esok.",
    "Diorang cakap server akan down untuk maintenance malam ni.",
]


def load_gold():
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    return gold


def extract_json_list(text: str):
    """Model output may have extra prose around the JSON list; pull out the first [...] block."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def main():
    gold = load_gold()

    print("Loading tokenizer/model from", MODEL_DIR)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, attn_implementation="eager")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s on {model.device}")

    results = []
    total_compared = 0
    total_match = 0

    for sentence in TEST_SENTENCES:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Lemmatize: {sentence}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        response = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        parsed = extract_json_list(response)

        print("\n" + "=" * 70)
        print("INPUT:", sentence)
        print("RAW OUTPUT:", response.strip()[:500])

        row = {"sentence": sentence, "raw_output": response, "parsed": parsed, "checks": []}

        if parsed is None:
            print("!! Could not parse JSON list from model output.")
        else:
            for item in parsed:
                if not isinstance(item, dict) or "surface" not in item or "lemma" not in item:
                    continue
                surface = str(item["surface"]).lower()
                lemma = str(item["lemma"]).lower()
                gold_lemma = gold.get(surface)
                if gold_lemma is None:
                    continue  # word not in gold dictionary, skip
                total_compared += 1
                match = lemma == gold_lemma.lower()
                total_match += match
                row["checks"].append({
                    "surface": surface, "model_lemma": lemma,
                    "gold_lemma": gold_lemma, "match": match,
                })
                mark = "OK" if match else "XX"
                print(f"  [{mark}] {surface!r:20} model={lemma!r:20} gold={gold_lemma!r}")

        results.append(row)

    print("\n" + "=" * 70)
    if total_compared:
        print(f"Word-level agreement with Malaya gold dictionary: "
              f"{total_match}/{total_compared} ({100 * total_match / total_compared:.1f}%)")
    else:
        print("No words could be matched against the gold dictionary "
              "(model output likely not valid JSON / not in expected format).")

    Path("test_lemmatizer_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Full results written to test_lemmatizer_results.json")


if __name__ == "__main__":
    main()
