# RU-RAD-6: Roman Urdu radicalization-related discourse markers

<!-- TODO after the Zenodo release: add the DOI badge here -->

Labels, annotation records, model outputs and code for the article

> **Measuring Radicalization-Related Discourse in Roman Urdu: Annotation Reliability, Label-Finalization Policy, and Supervised versus LLM Detection**. Syed Muhammad Waqas Shah, Humaira Arshad, Muhammad Ismail Mohmand and Usman Moyud Din Chaudhary. *Computers* (MDPI), submitted..

RU-RAD-6 contains 1,520 Roman Urdu social-media posts annotated for six discourse markers:
in-group/out-group framing, dehumanization, grievance, glorification of non-state violence,
mobilization and threat. Glorification is a pilot label; the benchmark compares models on the other five.

**Scope.** The labels describe textual expressions in individual posts. They do not identify
radicalized individuals, establish intent, or support enforcement decisions. The corpus is
purposively selected and enriched for rare markers, so label frequencies are not prevalence estimates.

**No post text is included.** Posts are keyed by `tweet_id = 'ru' + md5(normalized_text)[:10]`.
The text can be reconstructed from the two public source datasets (see "Reconstructing the text").

## Contents

| Folder | What it holds |
|---|---|
| `data/` | Released labels (`roman_urdu_indicators_v1.csv`), frozen partition (`splits.csv`: 1,099 train / 194 validation / 227 test), five-marker training view, label distribution, datasheet, pool counts and selection membership |
| `guidelines/` | Annotation guidelines v1.1 FINAL, the v1.1 addendum, the correction note used before re-annotation of the 200-post batch, and the indicator taxonomy |
| `annotation/` | Per-annotator sheets without text, the canonical and mined disagreement log with adjudication rationales, the blind adjudication of the 36 test-set disagreements, and annotator profiles |
| `model_outputs/` | Per-post scores and predictions for TF-IDF and XLM-RoBERTa, the XLM-RoBERTa and MuRIL reruns with run records, and the raw outputs of the three prompted models |
| `results/` | Agreement by batch, alpha and kappa with bootstrap intervals, macro-F1 by reference policy, paired bootstrap of the supervised–LLM gap, and the repeated cross-validation |
| `code/` | Pool construction, mining, text reconstruction, baselines, LLM prompting and scoring |
| `docs/` | Qwen2.5-72B serving record and source-contribution counts |

Provenance groups in `data/roman_urdu_indicators_v1.csv` (`split` column):

| `split` | Posts | Annotation | Label rule |
|---|---|---|---|
| `iaa` (canonical) | 350 | two annotators | adjudicated |
| `mined_320` | 320 | two annotators | intersection (both positive) |
| `solo2_200_relabelled` | 200 | two annotators | intersection (both positive) |
| `solo1`, `solo2` | 325 + 325 | one annotator | that annotator's label; training partition only |

## Quick check without post text

```bash
pip install pandas scikit-learn
python code/08_verify_splits.py            # audits the frozen partition
python code/13_score_released_outputs.py   # five-indicator macro-F1 under each reference
```

Expected five-indicator macro-F1 on the 227 test posts:

| System | Released | Intersection | Union | Adjudicated (36 cells) |
|---|---|---|---|---|
| TF-IDF | 0.340 | 0.318 | 0.309 | 0.329 |
| XLM-RoBERTa | 0.376 | 0.348 | 0.330 | 0.362 |
| Qwen2.5-7B (taxonomy) | 0.207 | 0.198 | 0.267 | 0.245 |
| Llama-3.1-8B (taxonomy) | 0.220 | 0.221 | 0.276 | 0.265 |
| Qwen2.5-72B (taxonomy) | 0.267 | 0.258 | 0.314 | 0.293 |

## Reconstructing the text

Obtain the two source datasets under their own terms:

- RUHSOLD (Rizwan, Shakeel and Karim, EMNLP 2020): https://github.com/haroonshakeel/roman_urdu_hate_speech (MIT licence)
- RU-HSD-30K (Bilal et al., IEEE Access 2022): https://github.com/Bilal4209/RU-HSD-30K (publicly available; no licence is stated in the repository)

```bash
python code/rehydrate.py --ruhsold_dir RUHSOLD \
    --ru30k_csv "RUHSD30K/final 30,000 dataset_romanurdu.csv" \
    --labels data/roman_urdu_indicators_v1.csv --out roman_urdu_indicators_v1_TEXT.csv
```

Do not commit the reconstructed file; `.gitignore` excludes it.

## Code

| Script | Purpose | Needs text | Status |
|---|---|---|---|
| `01_build_pool.py` | Standardize the sources and build the 39,458-post pool | source datasets | reproduces the pool counts in `data/pool_step_counts.csv` |
| `02_active_learning_mining.py` | Character n-gram rankers used to mine rare markers | yes | as used |
| `rehydrate.py` | Join labels back to text | source datasets | as used |
| `04_train_xlmr_v2_colab.py` | XLM-RoBERTa with checkpoint selection (Routine A) | yes | does not reproduce the reported scores (macro-F1 0.296) |
| `10_xlmr_rerun_both_routines_colab.py` | XLM-RoBERTa Routines A and B | yes | Routine B reproduces the reported scores (macro-F1 0.376) |
| `09_tfidf_baseline_frozen_split.py` | TF-IDF baseline on the frozen split | yes | regenerated; reproduces the released predictions (macro-F1 0.340) |
| `11_muril_rerun_tuned_threshold.py` | MuRIL diagnostic rerun | yes | diagnostic only (macro-F1 0.086) |
| `12_repeated_cv_870_tfidf.py` | Five repeats of five-fold cross-validation on the 870 double-annotated posts | yes | as used |
| `06_llm_annotation.py` | Prompts, decoding and parsing for Qwen2.5-7B and Llama-3.1-8B | yes | as used; contains the exact prompts and the five demonstrations |
| `07_llm_72b_hosted.py` | Qwen2.5-72B through a hosted endpoint | yes | as used; see `docs/Qwen72B_inference_settings.md` |
| `08_verify_splits.py`, `13_score_released_outputs.py` | Checks that run on the released files | no | run as shown above |
| `extensions_not_run/` | Few-shot and cue-ablation script for larger models | yes | **not run**; results are not in the article |

Section and table numbers in some script docstrings refer to earlier drafts of the manuscript.

## Known limits of this release

- The original TF-IDF script and the files used to select the first 1,000 posts were not recovered. The TF-IDF baseline is a regenerated one, and the initial selection cannot be reconstructed exactly.
- Library versions of the original supervised runs were not recorded. Versions of the reruns are in the `run_info.json` files.
- The immutable revision of the hosted Qwen2.5-72B model was not recorded.
- The bootstrap and kappa analyses in `results/` are provided as result tables; `13_score_released_outputs.py` recomputes the point estimates but not the intervals.
- Records of the exploratory human review of 44 selected errors are described in the article's supplement and are not part of this repository.
- A few annotator notes quote one to three words of a post.

## Licence

Code: MIT (`LICENSE`). Labels, guidelines, annotation records, model outputs and results: CC BY 4.0 (`LICENSE-DATA`).
RUHSOLD is released under the MIT licence. RU-HSD-30K is publicly available without a stated licence. For this reason the post text of neither dataset is redistributed here.

## Citation

See `CITATION.cff`. Please also cite the two source datasets.
