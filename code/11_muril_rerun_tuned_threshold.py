# =====================================================================
# RU-RAD-6  MuRIL per-indicator detectors  -  RERUN WITH A TUNED THRESHOLD
# ---------------------------------------------------------------------
# Purpose (Item 6): replace the earlier MuRIL result, which ran at an untuned
# threshold and predicted nothing for four of five indicators.
#
# Protocol (identical in spirit to the TF-IDF and XLM-R baselines):
#   * train on the frozen TRAIN fold only (1,099 posts)
#   * choose the training epoch by validation average precision
#   * choose the decision threshold by validation F1 (grid 0.05-0.95, step 0.01,
#     ties keep the lowest threshold)
#   * apply that threshold, unchanged, to the frozen TEST fold (227 posts)
#   * export per-post scores for validation AND test, thresholds, and run info
#
# Colab: Runtime > Change runtime type > GPU (T4 is enough, about 15-25 min).
# Upload to the Colab file panel BEFORE running:
#     gold_corpus_final.csv   (text-bearing corpus: tweet_id, text, six labels)
#     splits.csv              (tweet_id, fold = train / val / test)
# Run all cells (or: !python muril_rerun_tuned_threshold.py).
# Send back the downloaded muril_rerun_outputs.zip.
# =====================================================================
import os, sys, json, copy, random, platform, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import transformers, sklearn
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

# ----------------------------- configuration -------------------------
SMOKE = os.environ.get("MURIL_SMOKE") == "1"        # tiny CPU test of the code path only
DATA_DIR = os.environ.get("DATA_DIR", ".")
OUT_DIR = os.environ.get("OUT_DIR", "muril_rerun_outputs")
MODEL = "google/muril-base-cased"
SEED = 42
EPOCHS = 2 if SMOKE else 10
LR, BATCH, EVAL_BATCH, MAX_LEN = 2e-5, 16, 64, 128
WEIGHT_DECAY, WARMUP = 0.01, 0.1
INDICATORS = ["ig_outgroup", "dehumanization", "grievance", "mobilization", "threat"]
GRID = np.round(np.arange(0.05, 0.9501, 0.01), 2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ----------------------------- tokenizer / model ---------------------
def load_tokenizer():
    if not SMOKE:
        return AutoTokenizer.from_pretrained(MODEL)
    from transformers import BertTokenizerFast                    # smoke test only
    chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'\"-_@#:/")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + ["##" + c for c in chars]
    path = os.path.join(OUT_DIR, "smoke_vocab.txt")
    open(path, "w").write("\n".join(vocab))
    return BertTokenizerFast(path, do_lower_case=True)


def load_model(tok):
    if not SMOKE:
        return AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    from transformers import BertConfig, BertForSequenceClassification
    cfg = BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64, num_labels=2)
    return BertForSequenceClassification(cfg)


# ----------------------------- data ----------------------------------
corpus = pd.read_csv(os.path.join(DATA_DIR, "gold_corpus_final.csv"))
splits = pd.read_csv(os.path.join(DATA_DIR, "splits.csv"))
df = corpus.merge(splits, on="tweet_id", how="inner")
sizes = df["fold"].value_counts().to_dict()
print("fold sizes:", sizes)
assert sizes == {"train": 1099, "val": 194, "test": 227}, "unexpected fold sizes - wrong files?"
assert df["text"].notna().all(), "text column missing values"
tr, va, te = (df[df.fold == f].reset_index(drop=True) for f in ("train", "val", "test"))
tok = load_tokenizer()


def encode(texts):
    return tok(list(texts), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")


enc = {"train": encode(tr.text), "val": encode(va.text), "test": encode(te.text)}


@torch.no_grad()
def score(model, e):
    model.eval(); out = []
    n = e["input_ids"].shape[0]
    for i in range(0, n, EVAL_BATCH):
        b = {k: v[i:i + EVAL_BATCH].to(DEVICE) for k, v in e.items()}
        out.append(torch.softmax(model(**b).logits, dim=1)[:, 1].float().cpu().numpy())
    return np.concatenate(out)


def best_threshold(y, p):
    best_t, best_f = None, -1.0
    for t in GRID:                                   # ascending grid; strict > keeps the lowest tie
        f = precision_recall_fscore_support(y, (p >= t).astype(int), average="binary",
                                            zero_division=0)[2]
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def train_indicator(ind):
    set_seed(SEED)
    ytr = tr[ind].astype(int).values
    n1 = int(ytr.sum()); n0 = len(ytr) - n1
    weights = torch.tensor([len(ytr) / (2 * n0), len(ytr) / (2 * n1)], dtype=torch.float).to(DEVICE)
    model = load_model(tok).to(DEVICE)
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=LR)
    steps = EPOCHS * int(np.ceil(len(ytr) / BATCH))
    sch = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    lossf = nn.CrossEntropyLoss(weight=weights)
    y_t = torch.tensor(ytr, dtype=torch.long)
    gen = torch.Generator().manual_seed(SEED)
    best = {"ap": -1.0, "epoch": 0, "state": None}
    for ep in range(1, EPOCHS + 1):
        model.train(); perm = torch.randperm(len(ytr), generator=gen); losses = []
        for i in range(0, len(ytr), BATCH):
            ix = perm[i:i + BATCH]
            b = {k: v[ix].to(DEVICE) for k, v in enc["train"].items()}
            loss = lossf(model(**b).logits, y_t[ix].to(DEVICE))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); losses.append(loss.item())
        pv = score(model, enc["val"])
        ap = average_precision_score(va[ind].astype(int).values, pv)
        print(f"   {ind:15s} epoch {ep:2d}  train loss {np.mean(losses):.4f}  val AP {ap:.3f}"
              f"  val max score {pv.max():.3f}")
        if ap > best["ap"]:                                   # strict > keeps the earliest best epoch
            best = {"ap": ap, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    pv, pt = score(model, enc["val"]), score(model, enc["test"])
    del model, opt; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return best, pv, pt, n1


# ----------------------------- run ------------------------------------
t0 = time.time(); long_rows = []; summary = []
for ind in INDICATORS:
    print(f"\n=== {ind} ===")
    best, pv, pt, n1 = train_indicator(ind)
    yv, yt = va[ind].astype(int).values, te[ind].astype(int).values
    thr, vf1 = best_threshold(yv, pv)
    pred_t = (pt >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(yt, pred_t, average="binary", zero_division=0)
    summary.append(dict(indicator=ind, train_pos=n1, val_pos=int(yv.sum()), test_pos=int(yt.sum()),
                        best_epoch=best["epoch"], val_ap=round(best["ap"], 4), threshold=thr,
                        val_f1_at_threshold=round(vf1, 4),
                        test_ap=round(average_precision_score(yt, pt), 4),
                        test_precision=round(p, 4), test_recall=round(r, 4), test_f1=round(f, 4),
                        test_predicted_pos=int(pred_t.sum()),
                        test_score_max=round(float(pt.max()), 4), test_score_mean=round(float(pt.mean()), 4)))
    print("  ", summary[-1])
    for fold, d, y, s in (("val", va, yv, pv), ("test", te, yt, pt)):
        for tid, yy, ss in zip(d.tweet_id, y, s):
            long_rows.append(dict(tweet_id=tid, fold=fold, indicator=ind, y_true=int(yy),
                                  y_score=round(float(ss), 6), threshold=thr,
                                  y_pred=int(ss >= thr)))

res = pd.DataFrame(summary)
res["macro_f1_five_indicators"] = np.nan                   # filled on the first row only
res.loc[0, "macro_f1_five_indicators"] = round(float(res.test_f1.mean()), 4)
res.to_csv(os.path.join(OUT_DIR, "muril_rerun_results.csv"), index=False)
pd.DataFrame(long_rows).to_csv(os.path.join(OUT_DIR, "muril_rerun_per_post_scores.csv"), index=False)
info = dict(model=MODEL, smoke_test=SMOKE, seed=SEED, epochs=EPOCHS, lr=LR, batch=BATCH,
            max_len=MAX_LEN, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP,
            loss="class-weighted cross-entropy (inverse frequency)",
            epoch_selection="validation average precision (earliest best epoch)",
            threshold_selection="validation F1, grid 0.05-0.95 step 0.01, lowest tie",
            python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
            sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__,
            device=DEVICE, gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            note="GPU kernels are not bit-deterministic; scores can differ in the last digits between runs.",
            minutes=round((time.time() - t0) / 60, 1))
json.dump(info, open(os.path.join(OUT_DIR, "run_info.json"), "w"), indent=2)
print("\n", res.to_string(index=False)); print(json.dumps(info, indent=2))

if not SMOKE:
    try:
        import shutil
        shutil.make_archive("muril_rerun_outputs", "zip", OUT_DIR)
        from google.colab import files
        files.download("muril_rerun_outputs.zip")
    except Exception as e:
        print("Download manually from the Colab file panel: muril_rerun_outputs.zip", e)
