"""Build sentence-level SFT training data for the rojak lemmatizer.

Reads cleaned/siraplimau_cleaned.jsonl (per-doc `sentences`) and
cleaned/word_lemma_dictionary.json (word -> lemma; identity fallback from the cleaning
pipeline, overlaid with BM_dict.csv's curated rojak-slang -> actual-Malay mappings), and
emits {"sentence": ..., "target": [{"surface":.., "lemma":..}, ...]} rows, split by document
into train/eval so near-duplicate sentences from one article don't leak across the split.

The full corpus yields ~250k filtered sentences -- far more than is practical to train an 8B
model on. Sentences containing a BM_dict slang word are prioritized (kept in full, up to a
cap) so slang coverage survives subsampling instead of being diluted away by the much larger
pool of ordinary sentences; the rest of the budget is filled with a random sample.
"""
import csv
import json
import random
import re
from pathlib import Path

CLEANED_JSONL = Path("cleaned/siraplimau_cleaned.jsonl")
GOLD_DICT = Path("cleaned/word_lemma_dictionary.json")
BM_DICT_CSV = Path("BM_dict.csv")
OUT_TRAIN = Path("cleaned/lemma_sft_train.jsonl")
OUT_EVAL = Path("cleaned/lemma_sft_eval.jsonl")

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*|\d+|[^\w\s]", re.UNICODE)

MIN_TOKENS = 4
MAX_TOKENS = 40
MIN_ALPHA_TOKENS = 3  # at least this many alphabetic tokens, else sentence is junk/boilerplate

RANDOM_STATE = 42
EVAL_DOC_FRACTION = 0.10

MAX_TRAIN_SENTENCES = 12000
MAX_EVAL_SENTENCES = 1500
MAX_SLANG_SHARE = 0.5  # cap slang-containing sentences at this fraction of each split's budget


def load_slang_words():
    with BM_DICT_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["rojak"].strip().lower() for row in reader if row.get("rojak")}


def tokenize(sentence: str):
    return TOKEN_RE.findall(sentence)


def build_target(tokens, gold: dict):
    target = []
    for tok in tokens:
        lemma = gold.get(tok.lower(), tok.lower() if re.search(r"[a-zA-Z]", tok) else tok)
        target.append({"surface": tok, "lemma": lemma})
    return target


def subsample(rows, budget, slang_flags, rows_slang_words, rng):
    """Stratified subsampling: guarantee at least one training example per distinct slang
    word (round-robin) instead of pure-random selection, which -- with ~350 distinct slang
    words competing for a few thousand slots -- can drop rare words' only examples entirely
    by chance. Remaining budget after covering every word once is filled with random
    additional slang sentences (repetition helps common words) then plain sentences."""
    if len(rows) <= budget:
        return rows

    slang_cap = int(budget * MAX_SLANG_SHARE)

    word_to_indices = {}
    for i, (is_slang, words) in enumerate(zip(slang_flags, rows_slang_words)):
        if not is_slang:
            continue
        for w in words:
            word_to_indices.setdefault(w, []).append(i)

    words_order = list(word_to_indices.keys())
    rng.shuffle(words_order)
    for w in words_order:
        rng.shuffle(word_to_indices[w])

    kept_slang_indices = []
    seen = set()
    # round 1: one sentence per distinct word, guaranteeing coverage
    for w in words_order:
        if len(kept_slang_indices) >= slang_cap:
            break
        for idx in word_to_indices[w]:
            if idx not in seen:
                kept_slang_indices.append(idx)
                seen.add(idx)
                break

    # round 2: fill remaining slang budget with more examples (repetition for common words)
    all_slang_indices = [i for i, is_slang in enumerate(slang_flags) if is_slang]
    rng.shuffle(all_slang_indices)
    for idx in all_slang_indices:
        if len(kept_slang_indices) >= slang_cap:
            break
        if idx not in seen:
            kept_slang_indices.append(idx)
            seen.add(idx)

    plain_indices = [i for i, is_slang in enumerate(slang_flags) if not is_slang]
    rng.shuffle(plain_indices)
    remaining = budget - len(kept_slang_indices)
    kept_plain_indices = plain_indices[:remaining]

    result_indices = kept_slang_indices + kept_plain_indices
    rng.shuffle(result_indices)
    print(f"  stratified: covered {min(len(words_order), slang_cap)}/{len(words_order)} "
          f"distinct slang words, {len(kept_slang_indices)} slang rows + "
          f"{len(kept_plain_indices)} plain rows")
    return [rows[i] for i in result_indices]


def main():
    gold = json.loads(GOLD_DICT.read_text(encoding="utf-8"))
    slang_words = load_slang_words()

    docs = [json.loads(line) for line in CLEANED_JSONL.read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(docs)} docs")

    rng = random.Random(RANDOM_STATE)
    doc_indices = list(range(len(docs)))
    rng.shuffle(doc_indices)
    n_eval_docs = max(1, int(len(doc_indices) * EVAL_DOC_FRACTION))
    eval_doc_set = set(doc_indices[:n_eval_docs])

    train_rows, eval_rows = [], []
    train_slang_flags, eval_slang_flags = [], []
    train_slang_words, eval_slang_words = [], []

    for i, doc in enumerate(docs):
        for sentence in doc.get("sentences", []):
            sentence = sentence.strip()
            if not sentence:
                continue
            tokens = tokenize(sentence)
            if len(tokens) < MIN_TOKENS or len(tokens) > MAX_TOKENS:
                continue
            n_alpha = sum(1 for t in tokens if re.search(r"[a-zA-Z]", t))
            if n_alpha < MIN_ALPHA_TOKENS:
                continue

            target = build_target(tokens, gold)
            row = {"sentence": sentence, "target": target}
            sentence_slang_words = {t.lower() for t in tokens if t.lower() in slang_words}
            is_slang = bool(sentence_slang_words)

            if i in eval_doc_set:
                eval_rows.append(row)
                eval_slang_flags.append(is_slang)
                eval_slang_words.append(sentence_slang_words)
            else:
                train_rows.append(row)
                train_slang_flags.append(is_slang)
                train_slang_words.append(sentence_slang_words)

    print(f"filtered: {len(train_rows)} train / {len(eval_rows)} eval sentences "
          f"({sum(train_slang_flags)} / {sum(eval_slang_flags)} contain a BM_dict slang word)")

    train_rows = subsample(train_rows, MAX_TRAIN_SENTENCES, train_slang_flags, train_slang_words, rng)
    eval_rows = subsample(eval_rows, MAX_EVAL_SENTENCES, eval_slang_flags, eval_slang_words, rng)

    with OUT_TRAIN.open("w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with OUT_EVAL.open("w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_train_slang = sum(1 for r in train_rows if any(t["surface"].lower() in slang_words for t in r["target"]))
    n_eval_slang = sum(1 for r in eval_rows if any(t["surface"].lower() in slang_words for t in r["target"]))
    print(f"train: {len(train_rows)} sentences ({n_train_slang} with slang) -> {OUT_TRAIN}")
    print(f"eval:  {len(eval_rows)} sentences ({n_eval_slang} with slang) -> {OUT_EVAL}")
    print("\nSample train row:")
    print(json.dumps(train_rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
