"""
llm_large_model.py — does a 70B model behave differently?

Why
---
The 7-8B evaluation found systematic over-production: Qwen2.5-7B assigned 3.5x
and Llama-3.1-8B 6.3x the gold positive count, with inter-model kappa of 0.242.
The obvious reviewer objection is that both models are small.

This runs the SAME prompt on the SAME 227-post test fold with a model roughly
ten times larger, via Hugging Face Inference Providers. If over-production
persists, the finding is about the construct rather than model capacity. If it
resolves, that is important to know before a reviewer discovers it.

Setup
-----
    pip install huggingface_hub pandas scikit-learn
    Token needs "Make calls to Inference Providers" permission.

Run
---
    python llm_large_model.py
"""
import os, re, json, time
import pandas as pd
from huggingface_hub import InferenceClient

# ─── paste your HF token here, keep the quotes ──────────────────────────
HF_TOKEN = "PASTE_YOUR_HF_TOKEN_HERE"
# ────────────────────────────────────────────────────────────────────────

# Primary choice: ~10x larger than the 8B already tested, same family as
# Llama-3.1-8B so the comparison isolates scale rather than family.
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# Fallbacks if that provider is unavailable:
#   MODEL = "Qwen/Qwen2.5-72B-Instruct"
#   MODEL = "meta-llama/Llama-3.1-70B-Instruct"

TAG = MODEL.split("/")[-1].replace("-Instruct", "").lower()

INDICATORS = ["ig_outgroup", "dehumanization", "grievance",
              "glorification", "mobilization", "threat"]
ID_COL, TEXT_COL = "tweet_id", "text"
OUT = f"{TAG}__taxonomy__testfold.csv"

# ─── definitions verbatim from the annotation guideline (unchanged) ─────
DEFS = {
 "ig_outgroup": "In-group vs out-group framing: divides the world into a virtuous 'us' and an opposed, illegitimate 'them' framed as in conflict. Includes calling a named group traitor/ghaddar/enemy, 'real [nation/faith]' exclusion tests, and an implied out-group ('they oppress us'). NOT ordinary we/us pride with no adversary, and NOT mere criticism of a policy or leader.",
 "dehumanization": "Denies human or moral status to a GROUP, or frames people as vermin, insects, animals, subhuman, disease, or evil-by-nature in a way that invites their exclusion or harm. NOT a single animal slur aimed at one person in a quarrel, and NOT in-group self-defence.",
 "grievance": "The in-group, OR a solidarity group the author identifies or sympathizes with, is framed as oppressed, wronged, deprived, or denied justice (collective victimhood). NOT personal complaints, NOT merely blaming or taunting a target without a victim frame, and NOT neutral news reporting of suffering.",
 "glorification": "Praises, celebrates, or morally justifies NON-STATE, sectarian, or militant violence or its actors, including martyrdom or heroism framing. NOT ordinary patriotic commemoration of the state armed forces, and NOT reporting or mourning violence without endorsing it.",
 "mobilization": "The post itself urges the audience to take or join contentious collective action: direct imperatives like 'come out', 'join', 'we must march', 'rise up', or a militant call to fight. NOT merely reporting or describing that a protest or struggle is happening.",
 "threat": "An explicit threat, warning of harm, or stated intent to harm a target: direct calls to harm, conditional warnings of harm ('or else you will be finished'), or revenge framing. NOT disagreement or criticism without intent to harm.",
}

HEADER = ("You are labeling short Roman Urdu (Latin-script Urdu, code-mixed) social "
          "media posts for six discourse indicators. A post may have zero, one, or "
          "several indicators. Most posts have NONE - only label an indicator when "
          "the rule clearly applies. Return ONLY a JSON object: "
          '{"labels": [<indicator names present>]}. Valid names: '
          + ", ".join(INDICATORS) + ".")

def build_prompt(t):
    d = "\n".join(f"- {k}: {v}" for k, v in DEFS.items())
    return f"{HEADER}\n\nIndicator definitions:\n{d}\n\nPost: {t}\nJSON:"

REFUSAL = re.compile(r"\b(i can(?:not|'t)|i am unable|i'm unable|as an ai|"
                     r"i won'?t|cannot assist|can't help|not able to)\b", re.I)

def parse_labels(raw):
    if not raw or not raw.strip():
        return [], "malformed"
    if REFUSAL.search(raw) and "{" not in raw:
        return [], "refusal"
    m = re.search(r'\{.*\}', raw, re.S)
    if not m:
        f = [i for i in INDICATORS if re.search(rf'\b{re.escape(i)}\b', raw)]
        return (f, "ok") if f else ([], "malformed")
    try:
        return [l for l in json.loads(m.group(0)).get("labels", [])
                if l in INDICATORS], "ok"
    except Exception:
        f = [i for i in INDICATORS if re.search(rf'\b{re.escape(i)}\b', m.group(0))]
        return (f, "ok") if f else ([], "malformed")


def main():
    if HF_TOKEN.startswith("PASTE"):
        raise SystemExit("Paste your Hugging Face token into HF_TOKEN first.")

    gold = pd.read_csv("roman_urdu_indicators_v1.csv")
    sp   = pd.read_csv("splits.csv")
    test = set(sp.loc[sp.fold == "test", ID_COL])
    df = gold[gold[ID_COL].isin(test)].reset_index(drop=True)
    assert len(df) == 227, f"expected 227 test posts, got {len(df)}"
    print(f"model: {MODEL}\ntest fold: {len(df)} posts\n")

    client = InferenceClient(api_key=HF_TOKEN)

    # resume support — reruns skip what is already done
    done = {}
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = {r[ID_COL]: r for _, r in prev.iterrows()}
        print(f"resuming: {len(done)} already scored\n")

    rows = list(done.values())
    for n, (_, r) in enumerate(df.iterrows(), 1):
        if r[ID_COL] in done:
            continue
        try:
            resp = client.chat_completion(
                model=MODEL,
                messages=[{"role": "user",
                           "content": build_prompt(str(r[TEXT_COL]))}],
                max_tokens=64, temperature=0.0)
            raw = resp.choices[0].message.content
        except Exception as ex:
            msg = str(ex)
            print(f"  [{n}] error: {msg[:110]}")
            if any(k in msg.lower() for k in ("quota", "credit", "limit", "402")):
                print("\n  credits exhausted — progress saved, rerun later")
                break
            time.sleep(3)
            continue

        labs, status = parse_labels(raw)
        row = {ID_COL: r[ID_COL], "raw": raw, "status": status}
        for ind in INDICATORS:
            row[ind] = int(ind in labs)
        rows.append(row)

        if n % 10 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"  {n}/{len(df)}  labels so far: "
                  f"{sum(sum(x[i] for i in INDICATORS) for x in rows)}")
        time.sleep(0.3)

    pred = pd.DataFrame(rows)
    pred.to_csv(OUT, index=False)
    print(f"\nsaved: {OUT}  ({len(pred)}/227 scored)")
    if len(pred) < 227:
        print("incomplete — rerun to continue")
        return
    evaluate(df, pred)


def evaluate(df, pred):
    from sklearn.metrics import precision_recall_fscore_support
    m = df[[ID_COL] + INDICATORS].merge(
        pred[[ID_COL] + INDICATORS], on=ID_COL, suffixes=("_g", "_p"))

    print(f"\n{'='*64}\n{MODEL}\n{'='*64}")
    print(f"{'indicator':16s}{'gold':>6s}{'pred':>7s}{'ratio':>8s}"
          f"{'prec':>8s}{'rec':>7s}{'f1':>7s}")
    tot_g = tot_p = 0; f1s = []
    for ind in INDICATORS:
        y, yh = m[ind+"_g"], m[ind+"_p"]
        p, r, f1, _ = precision_recall_fscore_support(
            y, yh, average="binary", zero_division=0)
        g_, p_ = int(y.sum()), int(yh.sum())
        tot_g += g_; tot_p += p_; f1s.append(f1)
        print(f"{ind:16s}{g_:6d}{p_:7d}{p_/max(g_,1):8.1f}"
              f"{p:8.3f}{r:7.3f}{f1:7.3f}")
    print(f"{'TOTAL':16s}{tot_g:6d}{tot_p:7d}{tot_p/tot_g:8.1f}")
    print(f"\nmacro-F1: {sum(f1s)/len(f1s):.3f}")

    s = m[[i+"_p" for i in INDICATORS]].sum(axis=1)
    sg = m[[i+"_g" for i in INDICATORS]].sum(axis=1)
    print(f"mean labels/post: gold {sg.mean():.2f}  model {s.mean():.2f}")
    print(f"posts with no indicator: gold {(sg==0).mean():.0%}  "
          f"model {(s==0).mean():.0%}")
    print(f"refusals: {(pred.status=='refusal').mean():.1%}   "
          f"malformed: {(pred.status=='malformed').mean():.1%}")

    print(f"\n{'='*64}\nCOMPARISON WITH THE SMALLER MODELS\n{'='*64}")
    print(f"  Qwen2.5-7B      3.5x over-production, macro-F1 0.172")
    print(f"  Llama-3.1-8B    6.3x over-production, macro-F1 0.220")
    print(f"  {TAG:15s} {tot_p/tot_g:.1f}x over-production, "
          f"macro-F1 {sum(f1s)/len(f1s):.3f}")
    print()
    if tot_p / tot_g > 2.0:
        print("  Over-production PERSISTS at scale. The finding is about the")
        print("  construct, not model capacity. This closes the obvious")
        print("  reviewer objection.")
    else:
        print("  Over-production is substantially REDUCED at scale.")
        print("  The claim must be narrowed to smaller models. Report this")
        print("  honestly — it is better found now than by a reviewer.")


if __name__ == "__main__":
    main()
