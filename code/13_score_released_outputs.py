#!/usr/bin/env python3
"""
13_score_released_outputs.py -- recompute the headline test-set scores from the released files only.

No post text is needed. Scores the fixed predictions of the five principal systems on the 227 test posts
against the released reference and against the intersection, union, per-annotator and adjudicated references
(five-indicator macro-F1; glorification is excluded).

Run from the repository root:  python code/13_score_released_outputs.py
"""
import glob, pandas as pd
from sklearn.metrics import f1_score

IND = ["ig_outgroup", "dehumanization", "grievance", "mobilization", "threat"]
lab = pd.read_csv("data/roman_urdu_indicators_v1.csv").merge(pd.read_csv("data/splits.csv"), on="tweet_id")
test = lab[lab.fold == "test"].set_index("tweet_id")

def sheets(n):
    fs = sorted(glob.glob(f"annotation/original_sheets_no_text/*annotator{n}_no_text.csv"))
    d = pd.concat(pd.read_csv(f) for f in fs).drop_duplicates("tweet_id").set_index("tweet_id")
    return d.loc[test.index, IND].fillna(0).astype(int)

a1, a2 = sheets(1), sheets(2)
refs = {"Released": test[IND].astype(int), "Annotator 1": a1, "Annotator 2": a2,
        "Intersection": (a1 & a2), "Union": (a1 | a2)}
adj = refs["Released"].copy()
for r in pd.read_csv("annotation/test_fold_adjudication/adjudication_36_decisions_no_text.csv").itertuples():
    if r.indicator in IND:
        adj.loc[r.tweet_id, r.indicator] = int(str(r.decision).strip().lower().startswith("p"))
refs["Adjudicated (36 cells)"] = adj

preds = {}
sup = pd.read_csv("model_outputs/supervised/per_post_predictions_tfidf_xlmr.csv")
for name, key in (("TF-IDF", "tfidf"), ("XLM-R", "xlmr")):
    preds[name] = sup[sup.model == key].pivot(index="tweet_id", columns="indicator", values="y_pred").loc[test.index, IND]
for name, f in (("Qwen2.5-7B", "qwen2.5-7b"), ("Llama-3.1-8B", "llama3.1-8b"), ("Qwen2.5-72B", "qwen2.5-72b")):
    preds[name] = pd.read_csv(f"model_outputs/llm/{f}__taxonomy__testfold.csv").set_index("tweet_id").loc[test.index, IND]

out = pd.DataFrame({rn: {m: sum(f1_score(r[c], p[c], zero_division=0) for c in IND) / len(IND) for m, p in preds.items()}
                    for rn, r in refs.items()}).round(3)
print(out.to_string())
