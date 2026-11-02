# Measured results

All numbers below come from executed CPU training and frozen-checkpoint evaluation, not from the paper. The main model uses training seed 7. Its checkpoint was selected at RL episode 10 using validation reward; all 100 episodes completed and the final iterate is also retained.

Each training run used 100 episodes × 40 steps, 256 matching initialization transitions, 160 imitation updates per actor, and 400 RL updates per actor and critic. There are six independent actor–critic pairs.

## Main checkpoint: 20 held-out traffic seeds

| Policy | Mean reward | J / completed Mbit ↓ | Service success ↑ | Mean queue (Mbit/vehicle) ↓ |
| --- | ---: | ---: | ---: | ---: |
| Untrained neural policy | 0.295 | 0.476 | 77.90% | 13.11 |
| Matching imitation only | 0.557 | 0.294 | 85.30% | 7.42 |
| Selected RL policy | 0.887 | 0.265 | 99.19% | 5.19 |
| Final RL iterate | 0.881 | 0.269 | 99.19% | 5.19 |
| Random feasible | 0.367 | 0.386 | 79.89% | 10.38 |
| All local | -0.072 | 2.111 | 22.83% | 56.43 |
| MEC first, local fallback | 0.178 | 0.693 | 46.77% | 36.19 |
| Model-based matching | 0.980 | 0.209 | 99.28% | 5.19 |

The selected RL model uses **31.5% less energy per completed Mbit** than random feasible decisions. Its paired energy difference is -0.121 J/Mbit, with an approximate 95% interval [-0.132, -0.111] over the same 20 traffic seeds.

The model-based matching planner remains better on energy and reward. These results do not establish that the neural policy is optimal, faster than matching, better than the paper’s baselines, or better in real road networks. The implemented baselines are simple local, random, MEC-first and matching policies; the paper’s full LyMADDPG and MADDPG baselines were not reproduced.

## Independent training seeds

| Training seed | Trained reward | Trained J/completed Mbit | Service success | Recorded training time |
| --- | ---: | ---: | ---: | ---: |
| 7 | 0.887 | 0.265 | 99.19% | 15.5 s |
| 19 | 0.894 | 0.260 | 99.30% | 14.5 s |
| 41 | 0.888 | 0.264 | 99.34% | 15.3 s |

Training times are measurements on this cloud CPU for this small workload and exclude installation and held-out evaluation. Other hardware and larger configurations can take longer. The 20 test traffic seeds are reused across training seeds for paired comparison, not treated as 60 independent scenarios. Each run’s original metrics, weights, history and manifest are in `artifacts/demo`, `artifacts/seed19` and `artifacts/seed41`.

## How to interpret the metrics

- **Service success** counts attempted packets completed inside the modeled service/contact budget. It excludes backlog waiting time and is not an end-to-end deadline guarantee.
- **Energy per completed Mbit** divides all consumed modeled energy, including failed attempts, by successfully processed data. It prevents a low-work policy from looking efficient solely because it does little.
- **Queue size** is real, unbounded simulator backlog. The selected model’s mean queue growth is 0.050 Mbit/vehicle/slot over these short episodes; this is not a proof of strong stability.
- **Reward** combines a dimensionless energy reward and a per-vehicle service-feasibility bonus. It is a training signal, not an accuracy percentage.
- **Confidence intervals** in JSON/plots describe variability over synthetic traffic seeds for a fixed model, not universal performance guarantees.

## Validation evidence

- 13 automated tests passed: exact assignment against exhaustive search; resource and link feasibility; queue conservation; paired exogenous randomness; failure accounting; graph neighbor dependence; actor/critic and target updates; Bellman terminal masking; replay capacity; checkpoint/data round trips; and a real small training workflow.
- Three complete 100-episode training runs and held-out evaluation completed with finite metrics. Every trained model improved over its own untrained and imitation-only checkpoints.
- The live dashboard and offline replay were checked in Chromium, including policy switching, playback, workload/seed changes, responsive layout and error responses. The offline document was served as a local document and then tested with browser networking disabled because the managed browser blocks direct file navigation.

For paper fidelity and simulator assumptions, read [METHODOLOGY.md](METHODOLOGY.md). For a demonstration script and explanations, read [SHOWCASE.md](SHOWCASE.md).
