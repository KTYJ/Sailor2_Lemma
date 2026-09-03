# Thorough Guide: Building a Malay Rojak Lemmatiser Dataset

This guide provides a comprehensive pipeline to download, merge, and format the **Husein Zolkepli / Malaysia-AI** normalization and stemmer datasets. Following these steps will give you a clean `Input -> Target` dataset ready to train **Nano T5**, **Malay Llama**, or **Sailor2-8B**.

---

## 1. Directory Structure Setup

Before running scripts, create a clean workspace on your local machine or training environment:

```text
malay-lemmatiser/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── download_data.py
│   └── build_dataset.py
└── train.json (Final output file)
```

---

## 2. Step-by-Step Python Implementation

### Step 1: Download & Extract the Components (`src/download_data.py`)
This script uses live HTTP streams to pull the exact normalization dictionary pairs and underlying structural datasets without requiring a full repository clone.

```python
import os
import json
import requests

def download_file(url, save_path):
    print(f"Fetching: {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(response.json(), f, ensure_ascii=False, indent=2)
        print(f"Successfully saved to {save_path}")
    else:
        print(f"Failed to fetch {url}. Status code: {response.status_code}")

# Paths
os.makedirs("data/raw", exist_ok=True)

# Directly targeted endpoints from huseinzol05/malay-dataset
URLS = {
    "normalization_dict": "https://raw.githubusercontent.com/huseinzol05/malay-dataset/master/normalization/normalization/dictionary.json",
    "stemmer_news": "https://raw.githubusercontent.com/huseinzol05/malay-dataset/master/normalization/stemmer/zikit-news.json"
}

if __name__ == "__main__":
    download_file(URLS["normalization_dict"], "data/raw/normalization_dict.json")
    download_file(URLS["stemmer_news"], "data/raw/stemmer_news.json")
```

### Step 2: Merge and Construct the Rojak Pipeline (`src/build_dataset.py`)
This script takes clean structural sentences, dynamically infuses casual text abbreviations using the normalization dictionary (to simulate real social media data), and creates structured `Instruction -> Input -> Target` schemas.

```python
import json
import random
import os

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def introduce_rojak_slang(sentence, norm_dict, swap_probability=0.4):
    """
    Reverses the normalization dictionary to inject chaotic internet slang
    back into a clean baseline text string.
    """
    words = sentence.split()
    rojak_words = []
    
    # Reverse lookup map: standard word -> list of slang variants
    reverse_dict = {}
    for slang, standard in norm_dict.items():
        if isinstance(standard, str):
            reverse_dict.setdefault(standard.lower(), []).append(slang)

    for word in words:
        clean_word = word.strip(".,!?()"").lower()
        if clean_word in reverse_dict and random.random() < swap_probability:
            slang_choice = random.choice(reverse_dict[clean_word])
            # Preserve punctuation if needed
            rojak_words.append(word.lower().replace(clean_word, slang_choice))
        else:
            rojak_words.append(word)
            
    return " ".join(rojak_words)

def build_dataset():
    norm_dict = load_json("data/raw/normalization_dict.json")
    
    # Husein's stemmer files are typically organized as pairs of [Original, Lemmatised]
    # Adjust structure checking depending on file composition
    stemmer_data = load_json("data/raw/stemmer_news.json")
    
    dataset = []
    
    for idx, item in enumerate(stemmer_data):
        # Handle implicit layout variations in raw JSON structures safely
        if isinstance(item, list) and len(item) >= 2:
            clean_text, lemma_text = item[0], item[1]
        elif isinstance(item, dict):
            clean_text = item.get("text") or item.get("original", "")
            lemma_text = item.get("lemma") or item.get("stemmed", "")
        else:
            continue
            
        if not clean_text or not lemma_text:
            continue
            
        # Synthesize real Rojak structure
        rojak_input = introduce_rojak_slang(clean_text, norm_dict)
        
        # Format explicitly for LLM text generation/Instruction tuning
        instruction_payload = {
            "instruction": "Lemmatise the following Malay Rojak text by normalizing abbreviations and isolating root words.",
            "input": rojak_input,
            "target": lemma_text
        }
        dataset.append(instruction_payload)
        
    # Output unified instruction set
    with open("train.json", 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Dataset completely built! Compiled {len(dataset)} instruction pairs in train.json")

if __name__ == "__main__":
    build_dataset()
```

---

## 3. Formatting Strategy for Each Model Candidate

Depending on which candidate architecture you feed `train.json` to, optimize your training pipeline inputs accordingly:

### A. Format Strategy for Nano T5
Nano T5 requires absolute, explicit mapping markers since it has no foundational zero-shot prompt adherence capabilities. Format your training tokens as straight string segments:
* **Input Text Stream:** `lemmatize: Sbb tu xleh bincang elok2`
* **Target Text Stream:** `sebab itu tidak boleh bincang elok`

### B. Format Strategy for Malay Llama & Sailor2-8B
Decoder-only models learn best through structural chat templates. Wrap your generated payload inside the **Alpaca** or **ChatML** configuration blocks:

```json
{
  "messages": [
    {"role": "system", "content": "You are a professional linguist specialising in Malaysian Bahasa Rojak morphology."},
    {"role": "user", "content": "Extract the lemmatised form of this sentence: Sbb tu xleh bincang elok2"},
    {"role": "assistant", "content": "sebab itu tidak boleh bincang elok"}
  ]
}
```

---

## 4. Crucial Rules for Post-Processing & Quality Filters
* **Strip Out Embedded URLs:** Social media datasets are riddled with raw web paths (`https://...`). Strip these patterns out via regex matching (`re.sub(r'http\S+', '', text)`) before feeding sentences into tokenizers.
* **Keep Multi-Lingual Inclusions:** Do not purge English root tokens. If a sentence contains words like `"mem-blur-kan"` or `"di-ignore"`, retain them! These exact boundaries teach the model the dynamic intersection point of regional Malay affix systems combined with international root phrases.
