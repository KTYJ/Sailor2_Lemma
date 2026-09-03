"""Quick demo of the trained rojak lemmatizer on fresh, hand-written sentences
(not from the training/eval sets)."""
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./trained_models/sailor2_malay_lemmatizer"

SYSTEM_PROMPT = (
    "You are a Malay-English (Rojak) linguistic lemmatizer. "
    "Given a sentence, return a JSON list of objects with 'surface' and 'lemma'. "
    "Preserve English root words and expand Malay slang/affixes."
)

DEMO_SENTENCES = [
    "Aku baru je update driver GPU aku, sekarang game lagi smooth gila.",
    "Korang nak squad up tak? Aku dah ready dari tadi.",
    "Boss level ni memang susah nak kena beat, dah cuba banyak kali still fail.",
    "Jom kita grinding sikit sebelum event ni tamat esok pagi.",
    "Wifi kat rumah aku slow gila, lag teruk time main online.",
]


def extract_json_list(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def main():
    print("Loading model from", MODEL_DIR)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, attn_implementation="eager")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s on {model.device}\n")

    for sentence in DEMO_SENTENCES:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Lemmatize: {sentence}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=300, do_sample=False, temperature=None, top_p=None,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        response = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = extract_json_list(response)

        print("=" * 70)
        print("INPUT: ", sentence)
        if parsed is None:
            print("!! could not parse output:", response.strip()[:300])
            continue
        surfaces = [str(p.get("surface", "")) for p in parsed if isinstance(p, dict)]
        lemmas = [str(p.get("lemma", "")) for p in parsed if isinstance(p, dict)]
        print("SURFACE:", " ".join(surfaces))
        print("LEMMA:  ", " ".join(lemmas))
        changed = [(s, l) for s, l in zip(surfaces, lemmas) if s.lower() != l.lower()]
        if changed:
            print("CHANGED:", ", ".join(f"{s}->{l}" for s, l in changed))
        print()


if __name__ == "__main__":
    main()
