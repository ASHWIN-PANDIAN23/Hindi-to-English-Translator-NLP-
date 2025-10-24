from typing import List, Tuple, Dict
import json
import numpy as np
from tqdm import tqdm

def compute_metrics(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    # BLEU, chrF, TER via sacrebleu
    import sacrebleu
    from nltk.translate.meteor_score import meteor_score
    from bert_score import score as bertscore

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    ter = sacrebleu.corpus_ter(hyps, [refs]).score

    # METEOR (average over sentences)
    import nltk

    meteor_vals = [meteor_score([refs[i].split()], hyps[i].split()) for i in range(len(hyps))]

    meteor = float(np.mean(meteor_vals))

    # BERTScore (default English model)
    P, R, F1 = bertscore(hyps, refs, lang="en", verbose=False)
    bert_p = float(P.mean().item())
    bert_r = float(R.mean().item())
    bert_f1 = float(F1.mean().item())

    return {
        "BLEU": float(bleu),
        "chrF": float(chrf),
        "TER": float(ter),
        "METEOR": float(meteor),
        "BERTScore_P": bert_p,
        "BERTScore_R": bert_r,
        "BERTScore_F1": bert_f1,
    }

def token_prf1(hyp_ids: List[List[int]], ref_ids: List[List[int]], pad_id: int) -> Dict[str, float]:
    # Simple token-level precision/recall/F1 ignoring PAD
    tp = 0; fp = 0; fn = 0
    for h, r in zip(hyp_ids, ref_ids):
        h = [t for t in h if t != pad_id]
        r = [t for t in r if t != pad_id]
        # multiset comparison (bag-of-tokens)
        from collections import Counter
        ch = Counter(h)
        cr = Counter(r)
        common = ch & cr
        tp += sum(common.values())
        fp += sum((ch - cr).values())
        fn += sum((cr - ch).values())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"Token_P": float(precision), "Token_R": float(recall), "Token_F1": float(f1)}