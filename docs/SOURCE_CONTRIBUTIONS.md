# Source contributions and pool reconstruction (computed, text-free)

Reproduced by running the repository's own `code/01_build_pool.py` unchanged on the public RUHSOLD release
(github.com/haroonshakeel/roman_urdu_hate_speech) and RU-HSD-30K (github.com/Bilal4209/RU-HSD-30K, 30,000 rows).

## 1. Pool (39,458 reproduced exactly)
| step                                                      |   posts |
|:----------------------------------------------------------|--------:|
| RUHSOLD coarse (task 1) after text de-dup                 |    9991 |
| RU-HSD-30K raw rows                                       |   30000 |
| Concatenated                                              |   39991 |
| After dropping texts < 8 characters (normalised)          |   39700 |
| After de-dup on normalised text = pool                    |   39458 |
| of which RUHSOLD                                          |    9983 |
| of which RU-HSD-30K                                       |   29475 |
| Pool posts passing keyword relevance filter (>=1 cluster) |    6677 |
| Pool posts with >=1 indicator-bearing cluster             |    3842 |

Attribution rule: the script concatenates RUHSOLD first, so a text present in both sources is counted as RUHSOLD.
`task_2_validation.tsv` as supplied is a byte-identical copy of `task_2_train.tsv`; this does not change the pool (de-duplicated on text).

## 2. Where the labelled posts come from
All 1,520 labelled posts are in the 39,458 pool (IDs are md5 of normalised text).

| Batch | Posts | RUHSOLD | RU-HSD-30K | Pass keyword filter |
|:--|--:|--:|--:|--:|
| iaa (canonical-350) | 350 | 88 | 262 | 339 |
| solo1 | 325 | 78 | 247 | 313 |
| solo2 | 325 | 90 | 235 | 308 |
| **Full-run 1,000** | **1,000** | **256** | **744** | **960** |
| mined_320 | 320 | 34 | 286 | 94 |
| solo2_200_relabelled | 200 | 16 | 184 | 77 |
| **All** | **1,520** | **306** | **1,214** | |

By frozen fold, posts in RUHSOLD: train 239/1,099, val 35/194, test 32/227.

## 3. What this does and does not show about selection
- Only 960 of the full-run 1,000 pass the keyword relevance filter (40 do not), so the 1,000 were **not** drawn solely from the keyword-filtered subset.
- The mined 320 and re-labelled 200 mostly fall outside it (94/320 and 77/200), consistent with model-based mining and re-labelling of earlier posts rather than keyword selection.
- The actual sampling procedure, seed and any oversampling of glorification candidates (planned in the v1.1 addendum, R5) are **not recorded** and cannot be recovered from these data. The paper should not describe the 1,000 as a keyword-filtered sample unless the author can confirm it.
