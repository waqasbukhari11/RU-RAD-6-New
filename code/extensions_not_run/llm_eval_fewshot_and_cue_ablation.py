# RU-RAD-6: LLM evaluation on the frozen 227-post test fold with
#   (a) taxonomy prompt, (b) taxonomy + few-shot from the TRAIN fold only,
#   each WITH and WITHOUT the cue "Most posts have NONE - only label ... clearly applies".
# Works with any OpenAI-compatible chat endpoint (hosted API, vLLM, Together, etc.).
#
#   pip install openai pandas
#   export BASE_URL=https://.../v1   API_KEY=...   MODEL=<provider model name>
#   python llm_eval_fewshot_and_cue_ablation.py
#   Optional: export CONDITIONS=fewshot_cue,fewshot_nocue   (default: all four)
# Inputs in DATA_DIR: gold_corpus_final.csv (with text), splits.csv (tweet_id, fold)
# Outputs: <tag>__<condition>.csv (tweet_id, raw, status, six 0/1 columns), run_info.json,
#          fewshot_examples_used.csv. Resumable. Temperature 0, one pass per post.
import os, json, re, time, platform, random
import pandas as pd
DATA_DIR = os.environ.get("DATA_DIR", "."); OUT = os.environ.get("OUT_DIR", "llm_out")
MODEL = os.environ.get("MODEL", "MODEL_NAME"); TAG = os.environ.get("TAG", re.sub(r"[^A-Za-z0-9.-]", "_", MODEL))
MOCK = os.environ.get("MOCK") == "1"; K_POS = int(os.environ.get("K_POS", 2)); K_NEG = int(os.environ.get("K_NEG", 4))
IND = ["ig_outgroup", "dehumanization", "grievance", "glorification", "mobilization", "threat"]
os.makedirs(OUT, exist_ok=True)
CUE = " Most posts have NONE - only label an indicator when the rule clearly applies."
HEAD = ("You are labeling short Roman Urdu (Latin-script Urdu, code-mixed) social media posts for six discourse indicators. "
        "A post may have zero, one, or several indicators.{cue} Return ONLY a JSON object: {{\"labels\": [<indicator names present>]}}. "
        "Valid indicator names: ig_outgroup, dehumanization, grievance, glorification, mobilization, threat.")
DEFS = """Indicator definitions:
- ig_outgroup: In-group vs out-group framing: divides the world into a virtuous 'us' and an opposed, illegitimate 'them' framed as in conflict. Includes calling a named group traitor/ghaddar/enemy, 'real [nation/faith]' exclusion tests, and an implied out-group ('they oppress us'). NOT ordinary we/us pride with no adversary, and NOT mere criticism of a policy or leader.
- dehumanization: Denies human or moral status to a GROUP, or frames people as vermin, insects, animals, subhuman, disease, or evil-by-nature in a way that invites their exclusion or harm. NOT a single animal slur aimed at one person in a quarrel, and NOT in-group self-defence.
- grievance: The in-group, OR a solidarity group the author identifies or sympathizes with, is framed as oppressed, wronged, deprived, or denied justice (collective victimhood). NOT personal complaints, NOT merely blaming or taunting a target without a victim frame, and NOT neutral news reporting of suffering.
- glorification: Praises, celebrates, or morally justifies NON-STATE, sectarian, or militant violence or its actors, including martyrdom or heroism framing. NOT ordinary patriotic commemoration of the state armed forces, and NOT reporting or mourning violence without endorsing it.
- mobilization: The post itself urges the audience to take or join contentious collective action: direct imperatives like 'come out', 'join', 'we must march', 'rise up', or a militant call to fight. NOT merely reporting or describing that a protest or struggle is happening.
- threat: An explicit threat, warning of harm, or stated intent to harm a target: direct calls to harm, conditional warnings of harm ('or else you will be finished'), or revenge framing. NOT disagreement or criticism without intent to harm."""

df = pd.read_csv(os.path.join(DATA_DIR, "gold_corpus_final.csv")).merge(pd.read_csv(os.path.join(DATA_DIR, "splits.csv")), on="tweet_id")
assert df.fold.value_counts().to_dict() == {"train": 1099, "val": 194, "test": 227}
train, test = df[df.fold == "train"], df[df.fold == "test"].reset_index(drop=True)

# few-shot examples: TRAIN fold only, fixed seed, K_POS positives per non-rare indicator + K_NEG all-negative posts
rng = random.Random(42); ex = []
for i in IND:
    pool = train[train[i] == 1].sort_values("tweet_id")
    ids = list(pool.tweet_id); rng.shuffle(ids)
    ex += [(t, i) for t in ids[:K_POS]]
neg = sorted(train[train[IND].sum(axis=1) == 0].tweet_id); rng.shuffle(neg); ex += [(t, "NEG") for t in neg[:K_NEG]]
seen, shots = set(), []
for t, why in ex:
    if t in seen: continue
    seen.add(t); r = train[train.tweet_id == t].iloc[0]
    shots.append((t, why, r.text, [i for i in IND if r[i] == 1]))
rng.shuffle(shots)
pd.DataFrame([dict(tweet_id=t, picked_for=w, labels="|".join(l)) for t, w, _, l in shots]).to_csv(os.path.join(OUT, "fewshot_examples_used.csv"), index=False)
shot_txt = "\n".join(f"Post: {tx}\nJSON: {json.dumps({'labels': l})}" for _, _, tx, l in shots)

def prompt(cond, post):
    cue = CUE if cond.endswith("_cue") else ""
    parts = [HEAD.format(cue=cue), "", DEFS]
    if cond.startswith("fewshot"): parts += ["", "Examples:", shot_txt]
    return "\n".join(parts) + f"\n\nPost: {post}\nJSON:"

def call(p):
    if MOCK: return json.dumps({"labels": []})
    from openai import OpenAI
    global _c
    try: _c
    except NameError: _c = OpenAI(base_url=os.environ["BASE_URL"], api_key=os.environ["API_KEY"])
    for a in range(4):
        try:
            return _c.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": p}], temperature=0, max_tokens=64).choices[0].message.content
        except Exception as e:
            time.sleep(2 ** a); err = str(e)
    return "ERROR: " + err

def parse(raw):
    m = re.search(r"\{.*?\}", raw or "", re.S)
    try: labs = json.loads(m.group(0))["labels"]; return "ok", {i: int(i in labs) for i in IND}
    except Exception: return "parse_fail", {i: 0 for i in IND}

CONDS = os.environ.get("CONDITIONS", "taxonomy_cue,taxonomy_nocue,fewshot_cue,fewshot_nocue").split(",")
for cond in CONDS:
    path = os.path.join(OUT, f"{TAG}__{cond}.csv"); done = set(pd.read_csv(path).tweet_id) if os.path.exists(path) else set()
    rows = []
    for _, r in test.iterrows():
        if r.tweet_id in done: continue
        raw = call(prompt(cond, r.text)); st, lab = parse(raw)
        rows.append(dict(tweet_id=r.tweet_id, raw=raw, status=st, **lab))
    if rows: pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print(cond, "done", len(rows), "new rows")
json.dump(dict(model=MODEL, tag=TAG, temperature=0, max_tokens=64, passes=1, k_pos=K_POS, k_neg=K_NEG, python=platform.python_version(),
               date=time.strftime("%Y-%m-%d"), base_url_host=re.sub(r"^https?://|/.*$", "", os.environ.get("BASE_URL", "")),
               note="record provider and exact model version string"), open(os.path.join(OUT, f"{TAG}_run_info.json"), "w"), indent=2)
