#!/usr/bin/env python3
"""
09_tfidf_baseline_frozen_split.py -- character TF-IDF + logistic regression on the frozen split.

Regenerated baseline (the original standalone script was not preserved; see Supplementary Methods S12).
Settings: character 2-5-grams, min_df=2, max 150,000 features, sublinear tf; class-balanced logistic
regression (liblinear); one binary classifier per indicator; features fitted on training text only;
threshold = value in a 0.01-step grid (0.05-0.95) that maximises validation F1, then fixed for the test set.

Needs post text: run code/rehydrate.py first (text is not redistributed).

Usage:
  python code/09_tfidf_baseline_frozen_split.py --corpus roman_urdu_indicators_v1_TEXT.csv \
      --splits data/splits.csv --out outputs/tfidf_per_post_predictions.csv
"""
import argparse, os, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

IND = ["ig_outgroup", "dehumanization", "grievance", "mobilization", "threat"]
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--out", default="outputs/tfidf_per_post_predictions.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.corpus)
    if "fold" not in df.columns:
        df = df.merge(pd.read_csv(a.splits), on="tweet_id")
    tr, va, te = (df[df.fold == f].reset_index(drop=True) for f in ("train", "val", "test"))
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=150000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr.text.astype(str)); Xva = vec.transform(va.text.astype(str)); Xte = vec.transform(te.text.astype(str))
    rows, f1s = [], []
    for c in IND:
        m = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=3000, random_state=42).fit(Xtr, tr[c].astype(int))
        pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        thr = max(GRID, key=lambda t: f1_score(va[c].astype(int), pv >= t, zero_division=0))
        pred = (pt >= thr).astype(int); f1 = f1_score(te[c].astype(int), pred, zero_division=0); f1s.append(f1)
        print(f"{c:15s} thr={thr:.3f} F1={f1:.3f}")
        rows += [dict(model="tfidf", indicator=c, tweet_id=i, y_true=int(y), y_score=float(s), threshold=float(thr), y_pred=int(p))
                 for i, y, s, p in zip(te.tweet_id, te[c], pt, pred)]
    print(f"five-indicator macro-F1 = {np.mean(f1s):.3f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); pd.DataFrame(rows).to_csv(a.out, index=False)

if __name__ == "__main__":
    main()
