import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm
import re
import unicodedata

def ensure_dirs(base_dir: Path):
    (base_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (base_dir / "models" / "vocab").mkdir(parents=True, exist_ok=True)
    (base_dir / "models" / "from_scratch").mkdir(parents=True, exist_ok=True)
    (base_dir / "results").mkdir(parents=True, exist_ok=True)

def normalize_text(text: str) -> str:
    # Basic Unicode normalization and whitespace clean-up
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def length_ratio_filter(src: str, tgt: str, max_ratio: float = 2.5) -> bool:
    # keep pair if max(len)/min(len) <= max_ratio (avoid ultra-unbalanced pairs)
    src_len = max(1, len(src.split()))
    tgt_len = max(1, len(tgt.split()))
    ratio = max(src_len, tgt_len) / max(1, min(src_len, tgt_len))
    return ratio <= max_ratio

def clean_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    cleaned = []
    for hi, en in pairs:
        hi_n = normalize_text(hi)
        en_n = normalize_text(en)
        if not hi_n or not en_n:
            continue
        if not length_ratio_filter(hi_n, en_n, max_ratio=2.5):
            continue
        cleaned.append((hi_n, en_n))
    return cleaned

def load_iitb() -> DatasetDict:
    # Loads the Hugging Face IITB dataset
    ds = load_dataset("cfilt/iitb-english-hindi")
    # ds has splits: train, test, validation with fields: 'translation': {'en', 'hi'}
    return ds

def ds_to_pairs(ds: Dataset) -> List[Tuple[str, str]]:
    pairs = []
    for ex in ds:
        d = ex.get("translation", {})
        hi = d.get("hi", "")
        en = d.get("en", "")
        pairs.append((hi, en))
    return pairs

def save_pairs_to_csv(pairs: List[Tuple[str,str]], path: Path):
    df = pd.DataFrame(pairs, columns=["hi", "en"])
    df.to_csv(path, index=False, encoding="utf-8")

def split_train_val_test(pairs: List[Tuple[str,str]],
                         val_ratio: float = 0.01,
                         test_ratio: float = 0.01,
                         seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n = len(idx)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test+n_val]
    train_idx = idx[n_test+n_val:]
    def subset(idxs):
        return [pairs[i] for i in idxs.tolist()]
    return subset(train_idx), subset(val_idx), subset(test_idx)