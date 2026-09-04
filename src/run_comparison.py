"""Run the section 3.1.4 evaluation-and-comparison table.

Scores the proposed fine-tuned Sailor2 lemmatiser against the baselines on one shared
held-out sample, with identical metrics and alignment (see eval_harness.py).

Examples
--------
    # everything the environment supports, 200 sentences, seed 7
    python run_comparison.py

    # just the rule-based baselines (works without torch)
    python run_comparison.py --systems identity,sastrawi

    # full run once torch + the fine-tuned Llama checkpoint exist
    python run_comparison.py --systems identity,malaya_naive,sastrawi,stem_lstm_512,sailor2,llama --n 300

Writes comparison_results.json (metrics + first-8 per-system examples) and prints the table.
"""
from __future__ import annotations

import argparse
import time
import traceback

import eval_harness as H
from lemmatizer_systems import ALL_SYSTEMS, build_system


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=",".join(ALL_SYSTEMS),
                    help="comma-separated system keys (default: all)")
    ap.add_argument("--n", type=int, default=H.DEFAULT_N, help="held-out sentences to score")
    ap.add_argument("--seed", type=int, default=H.DEFAULT_SEED, help="sampling seed")
    ap.add_argument("--out", default="results/comparison_results.json")
    ap.add_argument("--skip-missing", action="store_true", default=True,
                    help="skip a system whose dependencies fail to load (default on)")
    ap.add_argument("--no-skip-missing", dest="skip_missing", action="store_false")
    return ap.parse_args()


def main():
    args = parse_args()
    keys = [k for k in (s.strip() for s in args.systems.split(",")) if k]

    rows = H.load_eval_sample(n=args.n, seed=args.seed)
    n_tokens = sum(len(r["target"]) for r in rows)
    print(f"Held-out sample: {len(rows)} sentences / {n_tokens} gold tokens "
          f"(seed={args.seed}, from {H.EVAL_PATH})\n")

    results: dict[str, dict] = {}
    examples: dict[str, list] = {}
    skipped: dict[str, str] = {}

    for key in keys:
        print(f"=== {key} ===")
        try:
            system = build_system(key)
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            print(f"  skipped ({msg})\n")
            skipped[key] = msg
            if not args.skip_missing:
                raise
            continue

        t0 = time.time()
        try:
            preds = system.predict_batch(rows)
        except Exception:  # noqa: BLE001
            print(f"  FAILED during inference:\n{traceback.format_exc()}")
            skipped[key] = "inference error"
            if not args.skip_missing:
                raise
            continue

        metrics = H.score_predictions(rows, preds)
        metrics["runtime_sec"] = round(time.time() - t0, 1)
        results[system.name] = metrics

        ex = []
        for row, pred in list(zip(rows, preds))[:8]:
            aligned = H.align(pred, row["target"])
            ex.append({
                "sentence": row["sentence"],
                "tokens": [
                    {"surface": s, "gold": g, "pred": p,
                     "ok": (p is not None and p.strip().lower() == g.strip().lower())}
                    for s, g, p in aligned
                ],
            })
        examples[system.name] = ex

        print(f"  token_acc={metrics['token_accuracy']:.3f}  "
              f"lemma_acc={metrics['lemma_accuracy']:.3f}  "
              f"P/R/F1={metrics['precision']:.3f}/{metrics['recall']:.3f}/{metrics['f1']:.3f}  "
              f"cov={metrics['coverage']:.3f}  "
              f"unparse={metrics['unparseable_rate']:.3f}  "
              f"({metrics['runtime_sec']}s)\n")

    print("\n" + "=" * 78)
    print(f"COMPARISON  (n={len(rows)} sentences, {n_tokens} gold tokens, seed={args.seed})")
    print("=" * 78)
    if results:
        print(H.format_table(results))
    print("\nTokAcc = exact lemma / all gold tokens   LemAcc = exact lemma / gold-transformed tokens")
    print("Prec/Rec/F1 = detect-and-correct over transformed tokens   Cov = tokens predicted")
    if skipped:
        print("\nskipped:")
        for k, why in skipped.items():
            print(f"  {k}: {why}")

    H.write_results(
        args.out, results,
        meta={
            "n_sentences": len(rows), "n_gold_tokens": n_tokens, "seed": args.seed,
            "eval_path": str(H.EVAL_PATH), "skipped": skipped,
        },
        per_example=examples,
    )
    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()
