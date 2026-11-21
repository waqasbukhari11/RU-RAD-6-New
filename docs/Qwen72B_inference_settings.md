# Qwen2.5-72B-Instruct — Request Settings and Serving Version

**Prepared for:** supervisor / reproducibility record
**Paper:** RU-RAD-6 (violent-radicalization discourse; LLM annotation evaluation)
**Date of inference:** August 2026 (billing window 22–31 Aug 2026)

---

## Serving version / route
| Item | Value |
|---|---|
| Model | Qwen2.5-72B-Instruct |
| Access route | Hugging Face **Inference Providers** (hosted endpoint) |
| Upstream provider | **DeepInfra** |
| Served precision | **FP8** |
| Model revision | Provider-hosted "latest" at time of run (Aug 2026) |

*Billing confirms the provider: of 257 inference requests in the window, 248 were served by DeepInfra and 9 by Together AI (the 9 were initial tests, not part of the reported runs).*

## Request / decoding settings
| Setting | Value |
|---|---|
| Decoding | Greedy — `do_sample = False`, temperature 0; top-p not applied |
| Max new tokens | 64 |
| Repetition penalty | Not applied (library default 1.0) |
| Passes | Single deterministic pass |
| Output format | JSON object `{"labels": [...]}` |
| Prompt delivery | Single user turn via the model's chat template; no system prompt |

## Scope of the 72B run
| Item | Value |
|---|---|
| Prompt condition | **Taxonomy-guided only** (the six indicator definitions with exclusion clauses) |
| Evaluation set | Held-out **test fold, n = 227** |
| Requests | 227 (one per test-fold post) — consistent with the DeepInfra billing count |
| Parse outcome | All 227 responses parsed successfully (`status = ok`); **no refusals, no malformed responses** |

*(The prompt-sensitivity comparison across zero-shot / few-shot / taxonomy was run on Qwen2.5-7B-Instruct only; the 72B and Llama-3.1-8B were run under the taxonomy-guided prompt on the 227-post test fold for the cross-model comparison.)*

## Comparability note (for transparency)
The 7B and 8B models were served **locally** (Hugging Face transformers, 4-bit NF4 quantization, float16 compute, single NVIDIA T4). The 72B was served **remotely by DeepInfra at FP8**. Decoding was greedy and deterministic in all cases, but the 72B differs from the smaller models in serving stack and numerical precision (FP8 vs local 4-bit NF4). The 72B's responses were predominantly returned as Markdown-fenced JSON (a characteristic of the hosted chat-completions path); the parser extracted labels correctly in all 227 cases, so no data was lost. The cross-model (scale) contrast should therefore be read as indicative rather than precision-matched.
