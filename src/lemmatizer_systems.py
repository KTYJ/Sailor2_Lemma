"""Lemmatiser systems for the section 3.1.4 comparison, behind one interface.

Every system exposes:

    name        - short id used in the results table
    predict_batch(rows) -> list of predictions, one per row, each either a
                  [{"surface": str, "lemma": str}, ...] list or None (unparseable).

`rows` are the held-out records from eval_harness.load_eval_sample():
    {"sentence": str, "target": [{"surface","lemma"}, ...]}

The rule-based / seq2seq baselines are fed the GOLD surface tokens (row["target"] surfaces)
so tokenisation is identical to the gold and the only thing being measured is stem quality.
The generative LLM systems produce their own token list from the raw sentence, exactly as
they are used at inference time, and the harness aligns whatever they return.

Torch-backed systems import lazily so the rule-based path still runs when the torch install
is broken.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

_PUNCT_RE = re.compile(r"^\W+$", re.UNICODE)
# non-linguistic tokens Malaya's Base.stem() passes through untouched (url / email / @user /
# #tag / number / money / percent / time-ish). A pragmatic subset of malaya.text.regex.
_NONWORD_RE = re.compile(
    r"""^(
        https?://\S+ | www\.\S+ |            # url
        [\w.+-]+@[\w-]+\.[\w.-]+ |           # email
        [@#]\w+ |                            # mention / hashtag
        (?:rm|usd|\$)?\s?\d[\d.,:%]*\w* |    # number / money / percent / time
        \d{1,2}[:.]\d{2}
    )$""",
    re.IGNORECASE | re.VERBOSE,
)


def _apply_case(original: str, stemmed: str) -> str:
    """Re-impose the surface form's casing on the stem (Malaya's text.function.case_of)."""
    if not stemmed:
        return stemmed
    if original.isupper() and len(original) > 1:
        return stemmed.upper()
    if original[:1].isupper():
        return stemmed[:1].upper() + stemmed[1:]
    return stemmed


def _is_nonword(token: str) -> bool:
    s = token.strip()
    return not s or bool(_PUNCT_RE.match(s)) or bool(_NONWORD_RE.match(s))


def _load_english_words() -> set[str] | None:
    """Malaya's is_english() reference set (malaya.dictionary.ENGLISH_WORDS), if importable.

    Malaya's Naive/Sastrawi Base.stem() leaves any is_english() token unstemmed -- important
    for rojak text ('game', 'patch', 'lag'). Importing it pulls torch; when that fails the
    baselines still run, just without English pass-through (documented degraded mode).
    """
    try:
        from malaya.dictionary import ENGLISH_WORDS  # noqa: PLC0415

        return set(ENGLISH_WORDS)
    except Exception:  # noqa: BLE001
        return None


def _gold_surfaces(row) -> list[str]:
    return [t["surface"] for t in row["target"]]


# =========================================== Malaya affix stemmer, reimplemented

# malaya.text.tatabahasa.hujung  -- suffix key -> canonical form. Base.stem strips
# len(canonical) characters off the end when the surface *ends with* the key (so the noisy
# spellings 'kn'/'knn'/'nyaa' all collapse toward the 3-char canonical, sometimes eating a
# stem character -- faithful to the "naive" label).
_HUJUNG = {
    "kn": "kan", "knn": "kan", "kknn": "kan", "kkn": "kan", "kan": "kan", "kann": "kan",
    "kkann": "kan", "kaan": "kan", "kaann": "kan", "kah": "kah", "kahh": "kah",
    "lah": "lah", "lahh": "lah", "loh": "lah", "lohh": "lah", "lh": "lah", "lhh": "lah",
    "ler": "lah", "tah": "tah", "tahh": "tah", "nya": "nya", "nyaa": "nya", "nye": "nya",
    "nyee": "nya", "nyo": "nya", "nyoo": "nya", "ny": "nya", "an": "an", "ann": "an",
    "wan": "wan", "wann": "wan", "wati": "wati", "watii": "wati", "ita": "ita", "itaa": "ita",
}
# malaya.text.tatabahasa.permulaan  -- prefix key -> canonical form. Applied after the suffix
# cut, to the already-shortened word.
_PERMULAAN = {
    "bel": "bel", "se": "se", "see": "se", "ter": "ter", "terr": "ter", "men": "men",
    "menn": "men", "meng": "meng", "mengg": "meng", "mem": "mem", "mm": "mem",
    "memper": "memper", "di": "di", "ddi": "di", "pe": "pe", "ppe": "pe", "ppee": "pe",
    "me": "me", "mme": "mme", "ke": "ke", "kee": "ke", "ber": "ber", "berr": "ber",
    "pen": "pen", "penn": "pen", "per": "per", "perr": "perr",
}


def naive_stem_word(word: str) -> str:
    """Reimplementation of malaya.model.stem.Naive.stem_word.

    One suffix cut then one prefix cut; the number of characters removed is the length of the
    matched affix's *canonical* form, longest match wins; revert to the original if the word
    is fully consumed. No dictionary check on the result -- hence 'menarik' -> 'arik'.
    """
    original = word
    suffix_hits = [canon for key, canon in _HUJUNG.items() if word.endswith(key)]
    if suffix_hits:
        cut = max(suffix_hits, key=len)
        if cut:
            word = word[: -len(cut)]
    prefix_hits = [canon for key, canon in _PERMULAAN.items() if word.startswith(key)]
    if prefix_hits:
        cut = max(prefix_hits, key=len)
        if cut:
            word = word[len(cut):]
    return word or original


class _AffixStemSystem:
    """Token-wise stemmer over the gold surface tokens, with Malaya's Base.stem() pass-through
    (punctuation / numbers / urls / English words) and casing restoration."""

    def __init__(self):
        self._english = _load_english_words()
        self._cache: dict[str, str] = {}

    def stem_word(self, word: str) -> str:  # overridden by subclasses
        raise NotImplementedError

    def _stem(self, surface: str) -> str:
        if _is_nonword(surface):
            return surface.lower() if surface.strip() else surface
        low = surface.lower()
        if self._english is not None and low in self._english:
            return low
        if low not in self._cache:
            self._cache[low] = self.stem_word(low) or low
        return _apply_case(surface, self._cache[low]).lower()

    def predict_batch(self, rows):
        return [
            [{"surface": s, "lemma": self._stem(s)} for s in _gold_surfaces(r)]
            for r in rows
        ]


class IdentitySystem:
    """Lower bound: lemma == surface (lower-cased). Shows how much work the corpus needs."""

    name = "identity"

    def predict_batch(self, rows):
        return [
            [{"surface": s, "lemma": s.lower()} for s in _gold_surfaces(r)]
            for r in rows
        ]


class MalayaNaiveSystem(_AffixStemSystem):
    """malaya.stem.naive() reimplemented -- the paper's "Naive Stemmer from the Malaya library".

    Pure regex prefix/suffix stripping (naive_stem_word), no dictionary confirmation, so it
    over-stems ('menarik' -> 'arik'). Pure Python, no torch.
    """

    name = "malaya_naive"

    def stem_word(self, word: str) -> str:
        return naive_stem_word(word)


class SastrawiSystem(_AffixStemSystem):
    """malaya.stem.sastrawi() reimplemented: the PySastrawi confix-stripping stemmer applied
    per token, behind the same pass-through/casing wrapper Malaya uses. Includes lemmatisation
    (dictionary-confirmed roots), so it is far more conservative than the naive stemmer.
    Pure Python, no torch.
    """

    name = "sastrawi"

    def __init__(self):
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

        super().__init__()
        self._stemmer = StemmerFactory().create_stemmer()

    def stem_word(self, word: str) -> str:
        return self._stemmer.stem(word)


class StemLstm512System:
    """mesolitica/stem-lstm-512 -- the char-level LSTM seq2seq stemmer named in section 3.1.4.

    A trained model (malaya.torch_model.rnn.Stem), not reimplementable by hand; loaded via
    malaya.stem.huggingface(). `model` may also be 'mesolitica/stem-gru-bahdanau-1024'.
    Needs torch. Token-wise over the gold surfaces so tokenisation matches the other systems.
    """

    def __init__(self, model: str = "mesolitica/stem-lstm-512", beam_search: bool = False):
        import malaya

        self.name = model.split("/")[-1].replace("-", "_")
        print(f"[{self.name}] loading {model} via malaya.stem.huggingface ...")
        self._stemmer = malaya.stem.huggingface(model=model)
        self._beam_search = beam_search
        self._cache: dict[str, str] = {}

    def _stem(self, surface: str) -> str:
        if _is_nonword(surface):
            return surface.lower() if surface.strip() else surface
        low = surface.lower()
        if low not in self._cache:
            try:
                out = self._stemmer.stem(low, beam_search=self._beam_search)
            except TypeError:
                out = self._stemmer.stem(low)
            except Exception:  # noqa: BLE001
                out = low
            self._cache[low] = (out or low).strip()
        return self._cache[low]

    def predict_batch(self, rows):
        return [
            [{"surface": s, "lemma": self._stem(s)} for s in _gold_surfaces(r)]
            for r in rows
        ]


# ================================================================= generative LLMs

SYSTEM_PROMPT = (
    "You are a Malay-English (Rojak) linguistic lemmatizer. "
    "Given a sentence, return a JSON list of objects with 'surface' and 'lemma'. "
    "Preserve English root words and expand Malay slang/affixes."
)


def _extract_json_list(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class CausalLMSystem:
    """A merged causal-LM lemmatiser loaded from a local directory.

    Covers both the proposed fine-tuned Sailor2 model
    (trained_models/sailor2_malay_lemmatizer) and the fine-tuned Malaysian-Llama comparison
    model (trained_models/llama_malay_lemmatizer), since train_llama_lemmatizer_lora.py
    produces the identical prompt format and a merged checkpoint.
    """

    def __init__(self, model_dir: str, name: str, max_new_tokens: int = 400):
        self.name = name
        self._model_dir = model_dir
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[{self.name}] loading {self._model_dir} ...")
        t0 = time.time()
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_dir, dtype=torch.bfloat16, attn_implementation="eager"
        )
        self._model.to("cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval()
        print(f"[{self.name}] loaded in {time.time() - t0:.1f}s on {self._model.device}")

    def predict_batch(self, rows):
        if self._model is None:
            self._load()
        torch = self._torch
        preds = []
        t0 = time.time()
        for i, row in enumerate(rows):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Lemmatize: {row['sentence']}"},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                gen = self._model.generate(
                    **inputs, max_new_tokens=self._max_new_tokens, do_sample=False,
                    temperature=None, top_p=None,
                    eos_token_id=self._tokenizer.eos_token_id, repetition_penalty=1.1,
                )
            response = self._tokenizer.decode(
                gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            preds.append(_extract_json_list(response))
            if (i + 1) % 25 == 0:
                print(f"[{self.name}] {i + 1}/{len(rows)} "
                      f"({(time.time() - t0) / 60:.1f} min)")
        return preds


# ---------------------------------------------------------------- system registry

def build_system(key: str):
    """Map a --systems key to a constructed system. Raises if deps are missing."""
    key = key.strip().lower()
    if key == "identity":
        return IdentitySystem()
    if key == "sastrawi":
        return SastrawiSystem()
    if key in ("malaya_naive", "naive"):
        return MalayaNaiveSystem()
    if key in ("stem_lstm", "stem_lstm_512"):
        return StemLstm512System()
    if key in ("stem_gru", "stem_gru_bahdanau_1024"):
        return StemLstm512System(model="mesolitica/stem-gru-bahdanau-1024")
    if key in ("sailor2", "proposed"):
        return _causal_lm("./trained_models/sailor2_malay_lemmatizer", "sailor2_ft")
    if key in ("llama", "llama_ft"):
        return _causal_lm("./trained_models/llama_malay_lemmatizer", "llama_ft")
    if key in ("sailor2_base", "sailor2_zeroshot"):
        return CausalLMSystem("sail/Sailor2-3B-Chat", "sailor2_base")
    if key in ("llama_base", "llama_zeroshot"):
        return CausalLMSystem("mesolitica/Malaysian-Llama-3.2-3B-Instruct", "llama_base")
    raise KeyError(f"unknown system: {key!r}")


def _causal_lm(model_dir: str, name: str) -> "CausalLMSystem":
    if not (Path(model_dir) / "config.json").exists():
        raise FileNotFoundError(
            f"no merged checkpoint at {model_dir} "
            f"(run the matching train_*_lora.py first)"
        )
    return CausalLMSystem(model_dir, name)


ALL_SYSTEMS = [
    "identity", "malaya_naive", "sastrawi", "stem_lstm_512",
    "sailor2_base", "llama_base", "sailor2", "llama",
]
