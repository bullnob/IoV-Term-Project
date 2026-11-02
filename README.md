# Graph-assisted offloading for Internet of Vehicles

A working Python/PyTorch implementation inspired by **Algorithm 2 (CG-LyMADDPG)**
in Alam et al., *A Graph-Assisted Digital-Twin-Driven Multiagent Shared Offloading
for Internet of Vehicles*, IEEE Internet of Things Journal, 2025.
[DOI: 10.1109/JIOT.2025.3538657](https://doi.org/10.1109/JIOT.2025.3538657).

It includes an executable traffic/digital-twin simulator, exact capacitated
matching, a graph actor and centralized critic for **each vehicle**, actual
reinforcement learning, trained checkpoints, held-out evaluation, and an
interactive dashboard. No GPU, paid API, credentials, or dataset download is needed.

**Scope:** this is a documented research prototype, not the authors' source code
or a numerical reproduction of their figures. The supplied `Code.cpp` is a
frequency-counting program, not the paper's matching algorithm. It is preserved
unchanged in `references/supplied_Code.cpp`; our matching solver is implemented
independently. See [methodology and paper mapping](docs/METHODOLOGY.md).

## See the trained project immediately

Open **`artifacts/demo/showcase.html`** in a browser. It is self-contained and
works offline: switch between trained, untrained, random, all-local, MEC-first,
and matching policies; play or scrub the recorded decisions. No installation is
needed to present this file. It replays actual model-generated simulation results;
it does not train or run Python inside the browser.

For live inference in the prepared cloud workspace:

```bash
cd /workspace/IoV-Term-Project
source scripts/activate.sh
python -m iov.cli demo --run artifacts/demo --port 8000
```

Open the local server on port 8000 in your own machine's browser, or use your
development environment's supported port forwarding. In the live dashboard you
can change the traffic seed and arrival load. The static evaluation table always
describes the standard load; the top cards describe the selected live episode.

## Run on your computer

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a current
Python. The prepared environment uses **Python 3.15.2**, CPU PyTorch 2.15.0,
NumPy and SciPy. `uv.lock` records all dependencies. Then, in this repository:

```bash
uv sync --frozen
uv run python -m pytest -q
uv run python -m iov.cli demo --run artifacts/demo
```

The tested platform is Linux x86-64. Other platforms must have wheels available
on the configured PyTorch CPU index; their installation has not been tested.

## Train it yourself

The supplied checkpoints are already trained. To make a new run:

```bash
# In the cloud, activate scripts/activate.sh first. Else prefix commands with uv run.
python -m iov.cli train --config configs/demo.json --seed 7 --output runs/my-training
python -m iov.cli evaluate --run runs/my-training
python -m iov.cli export --run runs/my-training
python -m iov.cli demo --run runs/my-training
```

Training performs these operations automatically:

1. Run the matching planner in the simulator to collect 256 **transitions**.
2. Optionally warm-start actor networks with 160 matching-imitation updates.
3. Collect 4,000 exploratory RL transitions across 100 episodes. Each of the six
   actors and six critics receives 400 updates from replay, with soft target updates.
4. Evaluate periodically on six validation traffic seeds and save the best policy.
5. The separate evaluation command tests frozen models on 20 **different**, held-out
   traffic seeds, including untrained, imitation-only, final RL and reference policies.

Matching imitation is an explicitly labeled extension, not reinforcement learning.
Use `--no-imitation` to disable it. Matching rollouts still initialize replay as in
Algorithm 2. `configs/paper_training.json` uses the paper's reported learning rate
0.01 and soft-update coefficient 0.001; the tested demo uses 0.001 and 0.01 for
stable small-scale training. Neither preset claims to reconstruct every unspecified
paper parameter.

The console prints real validation metrics. `history.csv` is updated every episode.
An existing run directory is never overwritten. Checkpoints support inference;
they are **not** exact training-resume snapshots (optimizer/replay/RNG state is not saved).

## What to present

Read [the beginner's walkthrough and presentation script](docs/SHOWCASE.md), then
[the measured results](docs/RESULTS.md). The most useful deliverables are:

| File | Purpose |
| --- | --- |
| `artifacts/demo/showcase.html` | Portable, interactive presentation |
| `artifacts/demo/results.png` / `.svg` | Slide-ready learning and comparison charts |
| `artifacts/demo/best.pt` | Selected trained actor and critic weights |
| `artifacts/demo/matching_data.npz` | Matching-generated `(state, action, reward, next_state)` data |
| `artifacts/demo/evaluation.json` / `.csv` | Full held-out results and paired comparisons |
| `artifacts/demo/training_manifest.json` | Seeds, versions, update counts and training time |
| `artifacts/seed_summary.json` | Results across three independent training seeds |

## Source map

| Module | Responsibility |
| --- | --- |
| `src/iov/matching.py` | Capacity slots and exact maximum-weight bipartite assignment |
| `src/iov/environment.py` | Mobility, noisy twin estimates, energy, service deadlines and bit queues |
| `src/iov/networks.py` | Graph encoders, per-vehicle actors, centralized critics |
| `src/iov/learner.py` | Adam, Bellman targets, policy gradients, Lyapunov term and target updates |
| `src/iov/train.py` / `replay.py` | Rollouts, replay, matching warm-start and checkpoint selection |
| `src/iov/evaluation.py` / `report.py` | Paired held-out evaluation and plotted evidence |
| `src/iov/demo.py` / `web/index.html` | Live dashboard and offline export |
| `tests/` | Matching, queue conservation, learning, persistence and workflow checks |

