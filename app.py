"""Streamlit UI for the trained Malay-English rojak lemmatizer.

Supports multiple systems: Sailor2 (fine-tuned), Llama (fine-tuned), Malaya Naive,
Sastrawi, and stem-lstm-512. Includes a "Compare All" mode that runs every system
side-by-side on the same sentence.

Run locally with:
    .venv-1/Scripts/streamlit run app.py
"""
import time
import json
import re

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SAILOR2_DIR = "./trained_models/sailor2_malay_lemmatizer"
LLAMA_DIR   = "./trained_models/llama_malay_lemmatizer"

SAILOR2_BASE_ID = "sail/Sailor2-3B-Chat"
LLAMA_BASE_ID   = "mesolitica/Malaysian-Llama-3.2-3B-Instruct"

SYSTEM_PROMPT = (
    "You are a Malay-English (Rojak) linguistic lemmatizer. "
    "Given a sentence, return a JSON list of objects with 'surface' and 'lemma'. "
    "Preserve English root words and expand Malay slang/affixes."
)

EXAMPLE_SENTENCES = [
    "Game ni dah di-patch semalam tp still ngelag teruk.",
    "Kitorang xnak main sama dia sebab dia asyik toxic je.",
    "Korang nak pinjam kete aku ke petang ni?",
    "Jgn nti gaduh ngan ku, kita kena kerja sama-sama.",
]

SYSTEM_OPTIONS = {
    "🤖 Sailor2 (Fine-tuned LLM)": "sailor2",
    "🦙 Llama (Fine-tuned LLM)":   "llama",
    "🔤 Malaya Naive Stemmer":       "malaya_naive",
    "📚 Sastrawi Stemmer":           "sastrawi",
    "🧠 Stem LSTM-512 (Neural)":     "stem_lstm_512",
}

SYSTEM_DISPLAY = {
    "sailor2":       "Sailor2",
    "llama":         "Llama",
    "malaya_naive":  "Malaya",
    "sastrawi":      "Sastrawi",
    "stem_lstm_512": "LSTM-512",
}

st.set_page_config(page_title="Rojak Lemmatizer", page_icon="🇲🇾", layout="wide")


# ------------------------------------------------------------------ model loaders

@st.cache_resource(show_spinner="Loading Sailor2 model (first run only, ~10-20s)...")
def load_sailor2():
    tokenizer = AutoTokenizer.from_pretrained(SAILOR2_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        SAILOR2_DIR, dtype=torch.bfloat16, attn_implementation="eager"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner="Loading Llama model (first run only, ~10-20s)...")
def load_llama():
    import os
    if not os.path.isdir(LLAMA_DIR):
        return None, None, None
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_DIR, dtype=torch.bfloat16, attn_implementation="eager"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner="Loading base Sailor2 model (no fine-tuning, downloads on first run)...")
def load_sailor2_base():
    tokenizer = AutoTokenizer.from_pretrained(SAILOR2_BASE_ID)
    model = AutoModelForCausalLM.from_pretrained(
        SAILOR2_BASE_ID, dtype=torch.bfloat16, attn_implementation="eager"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner="Loading base Llama model (no fine-tuning, downloads on first run)...")
def load_llama_base():
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_BASE_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_BASE_ID, dtype=torch.bfloat16, attn_implementation="eager"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner="Loading Malaya Naive stemmer...")
def load_malaya_naive():
    from src.lemmatizer_systems import MalayaNaiveSystem
    return MalayaNaiveSystem()


@st.cache_resource(show_spinner="Loading Sastrawi stemmer...")
def load_sastrawi():
    from src.lemmatizer_systems import SastrawiSystem
    return SastrawiSystem()


@st.cache_resource(show_spinner="Loading Stem LSTM-512 (downloads on first run)...")
def load_stem_lstm():
    from src.lemmatizer_systems import StemLstm512System
    return StemLstm512System()


# ------------------------------------------------------------------ inference helpers

def extract_json_list(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _run_causal_lm(tokenizer, model, device, sentence: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Lemmatize: {sentence}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=400, do_sample=False, temperature=None, top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    elapsed = time.perf_counter() - t0
    response = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    parsed = extract_json_list(response)
    return parsed, elapsed, response


def run_sailor2(sentence: str):
    tokenizer, model, device = load_sailor2()
    return _run_causal_lm(tokenizer, model, device, sentence)


def run_llama(sentence: str):
    tokenizer, model, device = load_llama()
    if tokenizer is None:
        return None, 0.0, "[Llama model not found at trained_models/llama_malay_lemmatizer]"
    return _run_causal_lm(tokenizer, model, device, sentence)


def run_sailor2_base(sentence: str):
    tokenizer, model, device = load_sailor2_base()
    return _run_causal_lm(tokenizer, model, device, sentence)


def run_llama_base(sentence: str):
    tokenizer, model, device = load_llama_base()
    return _run_causal_lm(tokenizer, model, device, sentence)


def run_rule_based(sentence: str, system_key: str):
    """Run one of the rule-based / seq2seq systems on a raw sentence."""
    import re as _re
    TOKEN_RE = _re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*|\d+|[^\w\s]", _re.UNICODE)
    tokens = TOKEN_RE.findall(sentence)

    fake_row = {"sentence": sentence, "target": [{"surface": t, "lemma": t} for t in tokens]}

    if system_key == "malaya_naive":
        system = load_malaya_naive()
    elif system_key == "sastrawi":
        system = load_sastrawi()
    elif system_key == "stem_lstm_512":
        system = load_stem_lstm()
    else:
        raise ValueError(f"Unknown system: {system_key}")

    t0 = time.perf_counter()
    preds = system.predict_batch([fake_row])
    elapsed = time.perf_counter() - t0
    return preds[0], elapsed


def run_all_systems(sentence: str) -> dict:
    """Run all 5 systems, return dict keyed by system_key."""
    results = {}

    parsed, elapsed, raw = run_sailor2(sentence)
    rows = ([p for p in parsed if isinstance(p, dict) and "surface" in p and "lemma" in p]
            if parsed else [])
    results["sailor2"] = {"rows": rows, "elapsed": elapsed, "raw": raw}

    parsed, elapsed, raw = run_llama(sentence)
    rows = ([p for p in parsed if isinstance(p, dict) and "surface" in p and "lemma" in p]
            if parsed else [])
    results["llama"] = {"rows": rows, "elapsed": elapsed, "raw": raw}

    for key in ("malaya_naive", "sastrawi", "stem_lstm_512"):
        rows, elapsed = run_rule_based(sentence, key)
        results[key] = {"rows": rows or [], "elapsed": elapsed, "raw": None}

    return results


# ------------------------------------------------------------------ helpers for performance page

@st.cache_data
def load_comparison_results():
    import os
    path = "results/comparison_results.json"
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


SYSTEM_DISPLAY_ORDER = ["identity", "malaya_naive", "sastrawi", "stem_lstm_512", "sailor2_ft", "llama_ft"]
SYSTEM_PRETTY = {
    "identity":      "Identity (Baseline)",
    "malaya_naive":  "Malaya Naive",
    "sastrawi":      "Sastrawi",
    "stem_lstm_512": "Stem LSTM-512",
    "sailor2_ft":    "Sailor2 (Fine-tuned)",
    "llama_ft":      "Llama (Fine-tuned)",
}
METRIC_LABELS = {
    "token_accuracy":   "Token Acc.",
    "lemma_accuracy":   "Lemma Acc.",
    "precision":        "Precision",
    "recall":           "Recall",
    "f1":               "F1",
    "coverage":         "Coverage",
    "unparseable_rate": "Unparse Rate",
}


def render_performance_page(data: dict):
    results = data["results"]
    meta = data["meta"]

    st.subheader("Evaluation Summary")
    st.caption(
        f"n={meta['n_sentences']} sentences · {meta['n_gold_tokens']} gold tokens · seed={meta['seed']}"
    )
    with st.expander("📖 Metrics explained", expanded=False):
        st.markdown(
            """
| Metric | Definition |
|---|---|
| **Token Acc** | Exact lemma match / all gold tokens |
| **Lemma Acc** | Exact lemma match / only transformed tokens |
| **Precision** | Correct detections / all system detections (over transformed tokens) |
| **Recall** | Correct detections / all gold transformations (over transformed tokens) |
| **F1** | Harmonic mean of Precision & Recall |
| **Coverage** | Fraction of sentences fully parsed |
| **Unparse Rate** | Fraction of sentences with unparseable JSON output |
"""
        )

    # ---------- metrics table
    st.subheader("Metrics Table")
    metric_keys = ["token_accuracy", "lemma_accuracy", "precision", "recall", "f1", "coverage", "unparseable_rate"]
    systems_present = [s for s in SYSTEM_DISPLAY_ORDER if s in results]

    table = {"System": [SYSTEM_PRETTY.get(s, s) for s in systems_present]}
    for mk in metric_keys:
        table[METRIC_LABELS[mk]] = [round(results[s].get(mk, 0), 4) for s in systems_present]

    import pandas as pd
    df = pd.DataFrame(table)
    # Highlight best value per numeric column (excluding unparse rate where lower is better)
    def highlight_best(col):
        if col.name == "Unparse Rate":
            is_best = col == col.min()
        else:
            is_best = col == col.max()
        return ["background-color: #1a472a; color: white; font-weight: bold" if v else "" for v in is_best]

    numeric_cols = [METRIC_LABELS[mk] for mk in metric_keys]
    styled = df.style.apply(highlight_best, subset=numeric_cols)
    st.dataframe(styled, width="stretch", hide_index=True)

    st.divider()

    # ---------- bar charts
    st.subheader("Visual Comparison")
    chart_metrics = {
        "F1 Score (Detect & Correct)": "f1",
        "Lemma Accuracy": "lemma_accuracy",
        "Token Accuracy": "token_accuracy",
        "Precision": "precision",
        "Recall": "recall",
    }
    cols = st.columns(2)
    for i, (title, mk) in enumerate(chart_metrics.items()):
        chart_data = pd.DataFrame({
            "System": [SYSTEM_PRETTY.get(s, s) for s in systems_present],
            title: [results[s].get(mk, 0) for s in systems_present],
        })
        with cols[i % 2]:
            st.markdown(f"**{title}**")
            st.bar_chart(chart_data.set_index("System"), y=title, width="stretch")

    st.divider()

    # ---------- TP/FP/FN breakdown
    st.subheader("Error Breakdown (on transformed tokens)")
    err_table = {"System": [SYSTEM_PRETTY.get(s, s) for s in systems_present]}
    for col_name, key in [("TP (Correct)", "tp"), ("FP (Over-stemmed)", "fp"), ("FN (Missed)", "fn")]:
        err_table[col_name] = [results[s].get(key, 0) for s in systems_present]
    st.dataframe(pd.DataFrame(err_table), width="stretch", hide_index=True)

    # ---------- example sentences
    examples = data.get("examples", {})
    if examples:
        st.divider()
        st.subheader("Sample Predictions")
        example_system = st.selectbox(
            "Select system to inspect examples:",
            options=[s for s in systems_present if s in examples],
            format_func=lambda s: SYSTEM_PRETTY.get(s, s),
        )
        if example_system and example_system in examples:
            ex_list = examples[example_system][:5]
            for i, ex in enumerate(ex_list):
                with st.expander(f"Example {i+1}: {ex['sentence'][:80]}..."):
                    st.write(f"**Sentence:** {ex['sentence']}")
                    tokens = ex.get("tokens", [])
                    if tokens:
                        ex_df = pd.DataFrame([{
                            "Surface": t["surface"],
                            "Gold Lemma": t["gold"],
                            "Predicted": t["pred"],
                            "Correct": "✅" if t["ok"] else "❌",
                        } for t in tokens])
                        st.dataframe(ex_df, width="stretch", hide_index=True)


# ------------------------------------------------------------------ UI

st.title("🇲🇾 Malay-English Rojak Lemmatizer")
st.caption(
    "Benchmark multiple lemmatizer systems — Fine-tuned Sailor2-3B, Fine-tuned Llama-3.2-3B, "
    "Malaya Naive Stemmer, Sastrawi, and Stem LSTM-512."
)

if not torch.cuda.is_available():
    st.warning("No CUDA GPU detected — LLM inference will be slow on CPU.")

tab_lemmatizer, tab_performance = st.tabs(["🔤 Lemmatizer", "📊 Model Performance"])

with tab_performance:
    perf_data = load_comparison_results()
    if perf_data is None:
        st.warning("results/comparison_results.json not found. Run run_comparison.py to generate it.")
    else:
        render_performance_page(perf_data)

with tab_lemmatizer:
    st.subheader("Mode")
    mode = st.radio(
        "Mode",
        options=["Single System", "Compare All (side-by-side)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Single System":
        st.subheader("Select system")
        selected_label = st.radio(
            "Lemmatizer system",
            options=list(SYSTEM_OPTIONS.keys()),
            horizontal=True,
            label_visibility="collapsed",
        )
        selected_system = SYSTEM_OPTIONS[selected_label]

    st.divider()

    st.subheader("Try an example")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLE_SENTENCES):
        if cols[i % 2].button(ex, width="stretch"):
            st.session_state["sentence_input"] = ex

    sentence = st.text_area(
        "Or type your own sentence",
        key="sentence_input",
        height=100,
        placeholder="e.g. Kitorang xnak main sama dia sebab dia asyik toxic je.",
    )

    btn_label = "Compare All ▶" if mode == "Compare All (side-by-side)" else "Lemmatize ▶"

    if st.button(btn_label, type="primary", disabled=not sentence.strip()):

        # ---- COMPARE ALL ----
        if mode == "Compare All (side-by-side)":
            with st.spinner("Running all 5 systems..."):
                all_results = run_all_systems(sentence.strip())

            st.subheader("Comparison Results")

            timing_cols = st.columns(5)
            for i, key in enumerate(("sailor2", "llama", "malaya_naive", "sastrawi", "stem_lstm_512")):
                timing_cols[i].metric(SYSTEM_DISPLAY[key], f"{all_results[key]['elapsed']:.3f}s")

            st.divider()

            system_keys = list(SYSTEM_OPTIONS.values())
            cols = st.columns(len(system_keys))
            for col, key in zip(cols, system_keys):
                data = all_results[key]
                rows = data["rows"]
                with col:
                    st.markdown(f"**{SYSTEM_DISPLAY[key]}**")
                    if rows:
                        st.table({"Surface": [str(r["surface"]) for r in rows],
                                  "Lemma":   [str(r["lemma"])   for r in rows]})
                    else:
                        st.error("No output")
                        if data["raw"]:
                            with st.expander("Raw"):
                                st.code(data["raw"], language="text")

            st.divider()
            st.subheader("Cross-system token comparison")

            all_surfaces, seen = [], set()
            for key in system_keys:
                for r in all_results[key]["rows"]:
                    s = str(r["surface"])
                    if s.lower() not in seen:
                        seen.add(s.lower())
                        all_surfaces.append(s)

            if all_surfaces:
                lookup = {key: {str(r["surface"]).lower(): str(r["lemma"])
                                for r in all_results[key]["rows"]}
                          for key in system_keys}
                table = {"Surface": all_surfaces}
                for key in system_keys:
                    table[SYSTEM_DISPLAY[key]] = [lookup[key].get(s.lower(), "—") for s in all_surfaces]
                st.table(table)

        # ---- SINGLE SYSTEM ----
        else:
            with st.spinner(f"Running {selected_label}..."):
                if selected_system == "sailor2":
                    parsed, elapsed, raw_response = run_sailor2(sentence.strip())
                    rows = (
                        [p for p in parsed if isinstance(p, dict) and "surface" in p and "lemma" in p]
                        if parsed else []
                    )
                elif selected_system == "llama":
                    parsed, elapsed, raw_response = run_llama(sentence.strip())
                    rows = (
                        [p for p in parsed if isinstance(p, dict) and "surface" in p and "lemma" in p]
                        if parsed else []
                    )
                else:
                    result, elapsed = run_rule_based(sentence.strip(), selected_system)
                    rows = result if result else []
                    raw_response = None

            st.info(f"Inference time: **{elapsed:.3f}s**", icon="⏱")

            if not rows:
                st.error("No output produced. Raw response shown below.")
                if raw_response:
                    st.code(raw_response, language="text")
            else:
                surfaces = [str(r["surface"]) for r in rows]
                lemmas   = [str(r["lemma"])   for r in rows]

                st.subheader("Result")
                st.markdown(f"**Surface:** {' '.join(surfaces)}")
                st.markdown(f"**Lemma &nbsp;&nbsp;:** {' '.join(lemmas)}")

                changed = [(r["surface"], r["lemma"]) for r in rows if r["surface"].lower() != r["lemma"].lower()]
                if changed:
                    st.markdown("**Changed words:**")
                    st.table({"Surface": [c[0] for c in changed], "Lemma": [c[1] for c in changed]})
                else:
                    st.caption("No words were changed (already in base form, or unrecognized slang).")

                with st.expander("Full token table"):
                    st.table({"Surface": surfaces, "Lemma": lemmas})

                if raw_response is not None:
                    with st.expander(f"Raw model output ({selected_label})"):
                        st.code(raw_response, language="json")

    st.divider()
    st.subheader("🧪 Base Model Raw Output (No Fine-tuning)")
    st.caption(
        "Runs the un-tuned base checkpoints (same prompt, same generation settings as the "
        "fine-tuned models) so you can see exactly what they output before LoRA training — "
        "useful for inspecting why zero-shot accuracy/unparse-rate looks the way it does."
    )

    base_label = st.radio(
        "Base model",
        options=["🤖 Sailor2 (Base, no fine-tuning)", "🦙 Llama (Base, no fine-tuning)"],
        horizontal=True,
        key="base_model_choice",
    )

    base_sentence = st.text_area(
        "Sentence",
        key="base_sentence_input",
        height=100,
        placeholder="e.g. Kitorang xnak main sama dia sebab dia asyik toxic je.",
    )

    if st.button("Run base model ▶", type="secondary", disabled=not base_sentence.strip()):
        with st.spinner(f"Running {base_label}..."):
            if base_label.startswith("🤖"):
                parsed, elapsed, raw_response = run_sailor2_base(base_sentence.strip())
            else:
                parsed, elapsed, raw_response = run_llama_base(base_sentence.strip())

        st.info(f"Inference time: **{elapsed:.3f}s**", icon="⏱")

        st.markdown("**Raw model output (unparsed):**")
        st.code(raw_response, language="text")

        if parsed is None:
            st.error("Could not extract a valid JSON list from this output.")
        else:
            rows = [p for p in parsed if isinstance(p, dict) and "surface" in p and "lemma" in p]
            if rows:
                st.markdown("**Parsed result:**")
                st.table({"Surface": [str(r["surface"]) for r in rows],
                          "Lemma":   [str(r["lemma"])   for r in rows]})
            else:
                st.warning("JSON parsed but did not contain valid surface/lemma objects.")
