# =====================================================================
# RU-RAD-6  XLM-RoBERTa rerun: Routine A vs Routine B, compared with the
# existing per-post export.     (Run in Google Colab, GPU runtime.)
#
#   Routine A = the training script on record (04_train_xlmr_v2_colab_savescores.py):
#       HF Trainer, 15 epochs, lr 2e-5, batch 16/32, warm-up 0.1, weight decay 0.01,
#       weighted oversampling of positives, checkpoint chosen by validation AP.
#   Routine B = as described for Untitled48.ipynb: same, but 15 epochs and the FINAL
#       epoch is used (no checkpoint selection), no explicit warm-up / weight decay.
#       (Untitled48 itself was not available, so B is reconstructed from the
#        supplement's description; everything not described there is kept as in A.)
#
# Both: five separate binary models on the frozen train fold, threshold chosen on
# validation F1 over 37 equally spaced values 0.05-0.95 (lowest tie), applied to test.
#
# FILES to upload to the Colab file panel BEFORE running:
#   gold_corpus_final.csv   splits.csv       (text-bearing - do not publish)
#   per_post_predictions.csv                 (optional: the existing export, for comparison)
#
# STEP 1 (own cell, run first, then Runtime > Restart session):
#   !pip -q install transformers==4.44.2 datasets accelerate scikit-learn
# STEP 2: paste this whole file into one cell and run (about 1-1.5 h on a T4 for both routines).
# Set ROUTINES = ["A"] or ["B"] below to run only one.
# =====================================================================
import os, json, random, platform, time
import numpy as np, pandas as pd, torch
from torch.utils.data import WeightedRandomSampler
import transformers, sklearn, datasets as hfds
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

SMOKE = os.environ.get("XLMR_SMOKE") == "1"          # tiny CPU test of the code path only
ROUTINES = ["A", "B"]
DATA_DIR = os.environ.get("DATA_DIR", ".")
OUT_DIR = os.environ.get("OUT_DIR", "xlmr_rerun_outputs")
SEED, MODEL, MAXLEN, LR = 42, "xlm-roberta-base", 128, 2e-5
EPOCHS = 1 if SMOKE else 15
LABELS = ["ig_outgroup", "grievance", "mobilization", "threat", "dehumanization"]   # original order
GRID = np.linspace(0.05, 0.95, 37)
os.makedirs(OUT_DIR, exist_ok=True)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

corpus = pd.read_csv(os.path.join(DATA_DIR, "gold_corpus_final.csv"))
splits = pd.read_csv(os.path.join(DATA_DIR, "splits.csv"))
df = corpus.merge(splits, on="tweet_id", how="inner")            # keeps corpus row order (= train_ready.csv order)
assert df["fold"].value_counts().to_dict() == {"train": 1099, "val": 194, "test": 227}, "wrong files?"
tr, va, te = (df[df.fold == f].reset_index(drop=True) for f in ("train", "val", "test"))

if SMOKE:
    from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification
    chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'\"-_@#:/")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + ["##" + c for c in chars]
    vp = os.path.join(OUT_DIR, "smoke_vocab.txt"); open(vp, "w").write("\n".join(vocab))
    tok = BertTokenizerFast(vp, do_lower_case=True)
    def new_model():
        return BertForSequenceClassification(BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2,
                                             num_attention_heads=2, intermediate_size=64, num_labels=2))
    tr, va, te = tr.head(48), va.head(24), te.head(24)
else:
    tok = AutoTokenizer.from_pretrained(MODEL)
    def new_model():
        return AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

def make_ds(frame, col):
    e = tok(list(frame["text"].astype(str)), truncation=True, padding="max_length", max_length=MAXLEN)
    e["labels"] = frame[col].astype(int).tolist()
    return hfds.Dataset.from_dict(dict(e))

def ap_metric(eval_pred):
    logits, labels = eval_pred
    p = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    return {"auc_pr": average_precision_score(labels, p) if labels.sum() > 0 else 0.0}

class OverTrainer(Trainer):                                       # oversample positives (as in the script on record)
    def __init__(self, *a, sample_w=None, **k):
        super().__init__(*a, **k); self.sample_w = sample_w
    def _get_train_sampler(self, *args, **kwargs):
        return WeightedRandomSampler(self.sample_w, len(self.sample_w), replacement=True)

def pick_threshold(y, s):
    bt, bf = 0.5, -1.0
    for t in GRID:
        f = precision_recall_fscore_support(y, (s >= t).astype(int), average="binary", zero_division=0)[2]
        if f > bf: bf, bt = f, float(t)
    return bt

def run_routine(R):
    seed_all(SEED); long_rows = []; summ = []
    for c in LABELS:
        print(f"\n===== Routine {R} | {c} =====")
        y = tr[c].values.astype(int); w = np.where(y == 1, (len(y) - y.sum()) / max(y.sum(), 1), 1.0)
        kw = dict(output_dir=f"out_{R}_{c}", num_train_epochs=EPOCHS, learning_rate=LR,
                  per_device_train_batch_size=16, per_device_eval_batch_size=32,
                  seed=SEED, report_to="none", logging_steps=50)
        if R == "A":
            kw.update(eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
                      metric_for_best_model="auc_pr", greater_is_better=True, warmup_ratio=0.1, weight_decay=0.01)
        else:
            kw.update(eval_strategy="no", save_strategy="no")
        tra, vad, ted = make_ds(tr, c), make_ds(va, c), make_ds(te, c)
        try:
            targs = TrainingArguments(**kw)
        except TypeError:                      # newer transformers renamed warmup_ratio
            kw["warmup_steps"] = kw.pop("warmup_ratio"); targs = TrainingArguments(**kw)
        trn = OverTrainer(model=new_model(), args=targs, train_dataset=tra,
                          eval_dataset=vad if R == "A" else None,
                          compute_metrics=ap_metric if R == "A" else None,
                          sample_w=torch.tensor(w, dtype=torch.double))
        trn.train()
        sva = torch.softmax(torch.tensor(trn.predict(vad).predictions), 1)[:, 1].numpy()
        ste = torch.softmax(torch.tensor(trn.predict(ted).predictions), 1)[:, 1].numpy()
        yva, yte = va[c].values.astype(int), te[c].values.astype(int)
        thr = pick_threshold(yva, sva); pte = (ste >= thr).astype(int)
        P, Rc, F, _ = precision_recall_fscore_support(yte, pte, average="binary", zero_division=0)
        summ.append(dict(routine=R, indicator=c, threshold=round(thr, 4), test_ap=round(float(average_precision_score(yte, ste)), 4),
                         test_precision=round(P, 4), test_recall=round(Rc, 4), test_f1=round(F, 4),
                         predicted_pos=int(pte.sum()), reference_pos=int(yte.sum())))
        print(summ[-1])
        for fold, d, yy, ss in (("val", va, yva, sva), ("test", te, yte, ste)):
            for tid, a, b in zip(d.tweet_id, yy, ss):
                long_rows.append(dict(routine=R, tweet_id=tid, fold=fold, indicator=c, y_true=int(a),
                                      y_score=round(float(b), 6), threshold=thr, y_pred=int(b >= thr)))
        del trn; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    s = pd.DataFrame(summ); s["macro_f1"] = np.nan; s.loc[0, "macro_f1"] = round(float(s.test_f1.mean()), 4)
    return s, pd.DataFrame(long_rows)

t0 = time.time(); S, L = [], []
for R in ROUTINES:
    s, l = run_routine(R); S.append(s); L.append(l)
S = pd.concat(S); L = pd.concat(L)
S.to_csv(os.path.join(OUT_DIR, "xlmr_rerun_results.csv"), index=False)
L.to_csv(os.path.join(OUT_DIR, "xlmr_rerun_per_post_scores.csv"), index=False)

# ---- optional: quick comparison with the existing export -------------
cmp_rows = []
ex_path = os.path.join(DATA_DIR, "per_post_predictions.csv")
if os.path.exists(ex_path) and not SMOKE:
    ex = pd.read_csv(ex_path); ex = ex[ex.model == "xlmr"]
    for R in ROUTINES:
        for c in LABELS:
            a = L[(L.routine == R) & (L.fold == "test") & (L.indicator == c)].set_index("tweet_id")
            b = ex[ex.indicator == c].set_index("tweet_id").loc[a.index]
            cmp_rows.append(dict(routine=R, indicator=c,
                                 rerun_thr=a.threshold.iloc[0], export_thr=round(float(b.threshold.iloc[0]), 4),
                                 rerun_ap=round(float(average_precision_score(a.y_true, a.y_score)), 3),
                                 export_ap=round(float(average_precision_score(b.y_true, b.y_score)), 3),
                                 pred_agreement=round(float((a.y_pred.values == b.y_pred.values).mean()), 3),
                                 score_corr=round(float(np.corrcoef(a.y_score.values, b.y_score.values)[0, 1]), 3)))
    C = pd.DataFrame(cmp_rows); C.to_csv(os.path.join(OUT_DIR, "comparison_with_export.csv"), index=False)
    print("\n", C.to_string(index=False))
print("\n", S.to_string(index=False))
json.dump(dict(model=MODEL, smoke=SMOKE, seed=SEED, epochs=EPOCHS, lr=LR, maxlen=MAXLEN, routines=ROUTINES,
               python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
               datasets=hfds.__version__, sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__,
               device="cuda" if torch.cuda.is_available() else "cpu",
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               minutes=round((time.time() - t0) / 60, 1),
               note="B reconstructed from the supplement description; GPU kernels are not bit-deterministic."),
          open(os.path.join(OUT_DIR, "run_info.json"), "w"), indent=2)
if not SMOKE:
    try:
        import shutil
        shutil.make_archive("xlmr_rerun_outputs", "zip", OUT_DIR)
        from google.colab import files; files.download("xlmr_rerun_outputs.zip")
    except Exception as e:
        print("Download manually from the file panel: xlmr_rerun_outputs.zip", e)
