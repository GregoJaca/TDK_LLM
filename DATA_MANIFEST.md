# Data manifest (TDK_LLM)

Large data lives on `x11` under `/u0/grego/LLM`, not in git. See `chaos_llm_2026/docs/DATA_AND_CODE_MAP.md` for full details.

| Path | Size | Contents | Quality |
|---|---:|---|---|
| `~/LLM/TDK_LLM/runs_aug/results_perturbations` | 14 GB | DeepSeek single-token simplex hidden states, all layers, fp32 | Good for depth/layer analysis; no baseline |
| `~/LLM/TDK_LLM/runs_aug/results_perturbations_OLD` | 88 GB | DeepSeek legacy runs, radii 1e-8–1.0 | Same; many runs |
| `~/LLM/TDK_LLM/runs_aug/results_perturbations_microsoft` | 95 GB | Phi single-token simplex hidden states | Good for layer/depth, Phi |
| `~/LLM/TDK_LLM/runs_aug/results_jacobians*` | 294 MB | MLP/attention/normalization Jacobian metrics (JSON) | Metrics only; raw J not stored |
| `~/LLM/launch_sep` | 8.2 GB | 100 conditions × 2 prompts, last-layer hidden states | Dense Gaussian, no seed, repetition_penalty 1.1; not reproducible |
| `~/LLM/results_pecs` | ~4.6 GB | 12 prompts, first/middle/last hidden states, distance matrices | fp32 single trajectory; good for D2/RQA, no ensemble |
| `~/LLM/results_10_april` | ~4 GB | Temperature/radius trajectories, fractal/Lyapunov summaries | Legacy; dense Gaussian |
| `~/LLM/launch_aug` | ~2.5 GB | 4/16 conditions, layers -1/14/22 | Legacy; dense Gaussian, no seed |

Never commit `*.pt`, `*.npz`, `*.npy`, model directories, or the data above. The new fp32 A1/depth results live in `chaos_llm_2026/results/` and are ignored by `*.npz`.
