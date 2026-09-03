"""Streamlit UI for the trained Malay-English rojak lemmatizer.

Run locally with:
    .venv-1/Scripts/streamlit run app.py
"""
import json
import re

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./trained_models/sailor2_malay_lemmatizer"

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

st.set_page_config(page_title="Rojak Lemmatizer", page_icon="🇲🇾", layout="centered")


@st.cache_resource(show_spinner="Loading model (first run only, ~10-20s)...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, attn_implementation="eager"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def extract_json_list(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def lemmatize(sentence: str, tokenizer, model, device):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Lemmatize: {sentence}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=400, do_sample=False, temperature=None, top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    response = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response, extract_json_list(response)


st.title("🇲🇾 Malay-English Rojak Lemmatizer")
st.caption(
    "Fine-tuned Sailor2-3B-Chat (LoRA) — expands Malay slang/short forms and real "
    "morphological stems while preserving English loanwords. Running locally on your GPU."
)

if not torch.cuda.is_available():
    st.warning("No CUDA GPU detected — running on CPU will be slow for a 3B model.")

tokenizer, model, device = load_model()
st.success(f"Model loaded on **{device}**", icon="✅")

st.subheader("Try an example")
cols = st.columns(2)
for i, ex in enumerate(EXAMPLE_SENTENCES):
    if cols[i % 2].button(ex, use_container_width=True):
        st.session_state["sentence_input"] = ex

sentence = st.text_area(
    "Or type your own sentence",
    key="sentence_input",
    height=100,
    placeholder="e.g. Kitorang xnak main sama dia sebab dia asyik toxic je.",
)

if st.button("Lemmatize", type="primary", disabled=not sentence.strip()):
    with st.spinner("Running inference..."):
        raw_response, parsed = lemmatize(sentence.strip(), tokenizer, model, device)

    if parsed is None:
        st.error("Model output could not be parsed as JSON. Raw output shown below.")
        st.code(raw_response, language="text")
    else:
        rows = [p for p in parsed if isinstance(p, dict) and "surface" in p and "lemma" in p]
        surfaces = [str(r["surface"]) for r in rows]
        lemmas = [str(r["lemma"]) for r in rows]

        st.subheader("Result")
        st.markdown(f"**Surface:** {' '.join(surfaces)}")
        st.markdown(f"**Lemma:** {' '.join(lemmas)}")

        changed = [(r["surface"], r["lemma"]) for r in rows if r["surface"].lower() != r["lemma"].lower()]
        if changed:
            st.markdown("**Changed words:**")
            st.table({"Surface": [c[0] for c in changed], "Lemma": [c[1] for c in changed]})
        else:
            st.caption("No words were changed (already in base form, or unrecognized slang).")

        with st.expander("Full token table"):
            st.table({"Surface": surfaces, "Lemma": lemmas})

        with st.expander("Raw model output"):
            st.code(raw_response, language="json")
