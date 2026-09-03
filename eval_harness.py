"""Shared evaluation harness for the rojak lemmatiser comparison (test.ipynb section 3.1.4).

One held-out sample, one alignment routine, one metric set -- so the fine-tuned Sailor2
model and every baseline (Sastrawi, Malaya, stem-lstm-512, fine-tuned Malaysian-Llama) are
scored on exactly the same tokens with exactly the same rules.

Metrics (all computed against the gold {surface, lemma} targets in cleaned/lemma_sft_eval.jsonl):

  token_accuracy   - exact lemma match over ALL gold tokens. Tokens a system failed to
                     produce (unparseable output, dropped tokens) count as wrong. This is
                     the paper's "token-level accuracy".
  lemma_accuracy   - exact lemma match over only the gold-TRANSFORMED tokens (gold lemma
                     differs from the surface form). "When a word genuinely needs
                     lemmatising, how often is the lemma right." Punctuation / stopwords /
                     already-root words are excluded, so this is not inflated by the ~55%
                     of tokens that are trivially unchanged.
  precision/recall/f1 - lemmatisation treated as detect-and-correct over transformed tokens:
                     TP = system changed the token AND produced the correct lemma
                     FP = system changed the token but that was wrong (gold unchanged, or
                          wrong lemma)
                     FN = gold changed the token but the system did not (or got it wrong,
                          or never produced the token)
  coverage         - fraction of gold tokens the system produced an aligned prediction for
                     (1.0 for the rule-based baselines, <1.0 when the LLM drops/merges tokens)
  unparseable_rate - fraction of sentences whose model output could not be parsed to a
                     {surface, lemma} list (0.0 for baselines)
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

EVAL_PATH = Path("cleaned/lemma_sft_eval.jsonl")
DEFAULT_N = 200
DEFAULT_SEED = 7


# --------------------------------------------------------------------------- data

def load_eval_sample(n: int = DEFAULT_N, seed: int = DEFAULT_SEED, path: Path = EVAL_PATH):
    """Return a deterministic random sample of held-out rows: [{"sentence", "target"}, ...].

    `target` is the gold list of {"surface", "lemma"} dicts. Same seed => same sentences for
    every system, so results are directly comparable across runs.
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


# ---------------------------------------------------------------------- alignment

def _norm(s) -> str:
    return str(s).strip().lower()


def align(predicted, gold_target):
    """Align a predicted [{surface, lemma}] list against the gold target.

    Returns a list of (surface, gold_lemma, pred_lemma_or_None), one entry per gold token.

      * predicted is None (unparseable)         -> every pred_lemma is None
      * len(predicted) == len(gold)             -> align by position
      * otherwise                               -> greedy match on surface form; gold tokens
                                                   with no surviving predicted match get None
    """
    n_gold = len(gold_target)
    if predicted is None:
        return [(g["surface"], g["lemma"], None) for g in gold_target]

    pred_clean = [
        (str(it["surface"]), str(it["lemma"]))
        for it in predicted
        if isinstance(it, dict) and "surface" in it and "lemma" in it
    ]

    if len(pred_clean) == n_gold:
        return [
            (g["surface"], g["lemma"], p_lemma)
            for (_, p_lemma), g in zip(pred_clean, gold_target)
        ]

    by_surface: dict[str, list[str]] = {}
    for surf, lemma in pred_clean:
        by_surface.setdefault(_norm(surf), []).append(lemma)

    aligned = []
    for g in gold_target:
        cands = by_surface.get(_norm(g["surface"]))
        pred_lemma = cands.pop(0) if cands else None
        aligned.append((g["surface"], g["lemma"], pred_lemma))
    return aligned


# ------------------------------------------------------------------------ scoring

class Accumulator:
    """Accumulates per-token counts across a run and turns them into the metric dict."""

    def __init__(self) -> None:
        self.n_sent = 0
        self.n_unparseable = 0
        self.n_gold = 0
        self.n_aligned = 0            # gold tokens that got a (non-None) predicted lemma
        self.n_token_correct = 0     # exact lemma match, over all gold tokens
        self.n_gold_transformed = 0
        self.n_transformed_correct = 0   # correct lemma among gold-transformed tokens
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def add_sentence(self, predicted, gold_target) -> list[tuple]:
        self.n_sent += 1
        if predicted is None:
            self.n_unparseable += 1
        aligned = align(predicted, gold_target)
        for surface, gold_lemma, pred_lemma in aligned:
            self.n_gold += 1
            gold_change = _norm(gold_lemma) != _norm(surface)
            if gold_change:
                self.n_gold_transformed += 1

            if pred_lemma is None:
                # token never produced: wrong, and a miss if it should have changed
                if gold_change:
                    self.fn += 1
                continue

            self.n_aligned += 1
            correct = _norm(pred_lemma) == _norm(gold_lemma)
            pred_change = _norm(pred_lemma) != _norm(surface)

            if correct:
                self.n_token_correct += 1
                if gold_change:
                    self.n_transformed_correct += 1

            if pred_change and gold_change and correct:
                self.tp += 1
            elif pred_change:
                self.fp += 1          # changed it, but gold didn't or lemma is wrong
            elif gold_change:
                self.fn += 1          # left it alone, but gold changed it
        return aligned

    def metrics(self) -> dict:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "n_sentences": self.n_sent,
            "n_gold_tokens": self.n_gold,
            "token_accuracy": self.n_token_correct / self.n_gold if self.n_gold else 0.0,
            "lemma_accuracy": (
                self.n_transformed_correct / self.n_gold_transformed
                if self.n_gold_transformed else 0.0
            ),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "coverage": self.n_aligned / self.n_gold if self.n_gold else 0.0,
            "unparseable_rate": self.n_unparseable / self.n_sent if self.n_sent else 0.0,
            "n_gold_transformed": self.n_gold_transformed,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def score_predictions(rows, predictions) -> dict:
    """rows: [{"sentence","target"}]; predictions: list aligned to rows, each a
    [{"surface","lemma"}] list or None. Returns the metric dict."""
    acc = Accumulator()
    for row, pred in zip(rows, predictions):
        acc.add_sentence(pred, row["target"])
    return acc.metrics()


# -------------------------------------------------------------------- presentation

_COLS = [
    ("system", "System", "{:<28}"),
    ("token_accuracy", "TokAcc", "{:>7.3f}"),
    ("lemma_accuracy", "LemAcc", "{:>7.3f}"),
    ("precision", "Prec", "{:>6.3f}"),
    ("recall", "Rec", "{:>6.3f}"),
    ("f1", "F1", "{:>6.3f}"),
    ("coverage", "Cov", "{:>6.3f}"),
    ("unparseable_rate", "Unparse", "{:>8.3f}"),
]


def format_table(results_by_system: dict[str, dict]) -> str:
    """results_by_system: {system_name: metric_dict}. Returns a fixed-width text table."""
    widths = [28, 7, 7, 6, 6, 6, 6, 8]
    titles = [c[1] for c in _COLS]
    header = " ".join(t.rjust(w) if i else t.ljust(w) for i, (t, w) in enumerate(zip(titles, widths)))
    lines = [header, "-" * len(header)]
    for name, m in results_by_system.items():
        row = [name.ljust(widths[0])[:widths[0]]]
        for (key, _, _), w in list(zip(_COLS, widths))[1:]:
            row.append(f"{m.get(key, float('nan')):.3f}".rjust(w))
        lines.append(" ".join(row))
    return "\n".join(lines)


def write_results(path, results_by_system: dict, meta: dict | None = None,
                  per_example: dict | None = None) -> None:
    payload = {"meta": meta or {}, "results": results_by_system}
    if per_example:
        payload["examples"] = per_example
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
