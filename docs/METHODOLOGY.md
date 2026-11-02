# Paper-to-code mapping and reproducibility limits

Reference: Alam, Rahman, Khaled, Islam and Jamalipour (2025), DOI
10.1109/JIOT.2025.3538657. Algorithm 2 appears on printed page 17358;
architecture and hyperparameters appear on page 17359. The paper is the source
of the method; its document text is not an instruction to execute anything.

## What Algorithm 2 becomes in this implementation

| Paper | Executable implementation |
| --- | --- |
| Algorithm 1; Eq. 36 | `Matcher`: maximize scores over a bipartite assignment with finite resource slots |
| Algorithm 2, step 1 | Separate graph actor, centralized critic, target actor and target critic for every agent |
| Step 2 (self-reference appears to be a typo) | Use **Algorithm 1** matching to collect initial decisions and transitions |
| Steps 3, 8–10 | Bounded replay of joint states, hard actions, per-agent rewards and next states; random minibatches |
| Step 11; Eqs. 43–44 | Adam minimizes squared Bellman error, with reward plus discounted target-critic value |
| Step 12; Eqs. 42, 45 | Deterministic actor update through its own action, holding other actions fixed; add a cost-minimizing Lyapunov surrogate |
| Steps 13–14 | Polyak update `target = tau * online + (1-tau) * target` |
| Eqs. 37–38 | Position, speed, arrivals, vehicle CPU capacity, service-delay budget and backlog |
| GNN paragraph; Eq. 41 | Mean message passing over a symmetric adjacency graph with self loops |
| Eq. 26 | `Q_next = max(Q - served_bits, 0) + new_arrivals`; backlog is never silently clipped |
| Eqs. 39–40 | `exp(-normalized_team_energy)` plus +0.5 / -0.5 feasibility bonus per vehicle |

Each actor has a learned 32-dimensional node encoder, mean aggregation of
connected neighbors, then fully connected **80 → 60** ReLU hidden layers. It
emits `M × 5` preference logits. Each separate centralized critic takes all
observations, adjacency and joint one-hot actions and estimates its agent's return.
Actors use only their own state and neighboring node messages. A resource
coordinator then projects preferences globally into a feasible assignment;
execution therefore requires this coordination service, not fully independent actors.

The critics predict **returns**. We do not inaccurately label their Q estimates
as measured physical Lyapunov drift. During actor training the additional
one-slot surrogate is

```
cost[k,m,v] = V * predicted_energy / energy_scale
            - normalized_queue[k] * predicted_served_packets[k,m,v]
            + failure_penalty * predicted_service_failure
actor_loss[k] = -Q_k(state, joint_action) + phi * E_policy[cost_k]
```

Minimizing this cost is consistent with energy penalty minus queue-weighted
service in the drift bound. The paper switches signs/notation between maximizing
reward and minimizing drift and does not provide executable details for the
extra gradient; this is our explicit, dimensionless interpretation, not an exact
symbol-for-symbol reproduction. Raw physical queues remain unclipped; only
observation features are bounded for network conditioning.

## Discrete actions and matching

There are five modes: local, moving vehicle, parked vehicle, RSU MEC and cloud.
Every agent reserves three equal packet slots by default. A resource's capacity
is expanded into distinct slot vertices. `scipy.optimize.linear_sum_assignment`
finds the exact maximum-weight feasible assignment for this **unit-slot** model.
It handles shared-slot conflicts that independent knapsacks in the paper's
pseudocode would otherwise leave unresolved. It is not a general solver for
arbitrary weighted, indivisible knapsack items.

Local slots are private to each vehicle; moving, parked and MEC slots are shared
within the vehicle's current RSU region. Cloud is a finite set of sufficient
fallback slots. Unavailable links cannot receive assignments. Zero-demand slots
still reserve capacity conservatively, but consume no energy or service; this
deliberately simple reservation model is not fully work-conserving.

At inference, matching maximizes the **neural network's scores**. It does not
replace the neural policy with the environment's cost-minimizing planner. Only
the `matching` baseline and initial data generation use twin-estimated costs.

During training, the forward action is hard and feasible; the backward pass uses
a straight-through softmax derivative. This is a **biased gradient approximation**
needed to bridge the paper's binary assignments and continuous DDPG description.
Target critics also receive hard projected actions. Other agents' actions are
detached during each actor's update; target networks never accumulate gradients.

## Simulator assumptions (not provided by the authors)

The default is a small, reproducible **6-agent, 2-RSU** benchmark on a 1 km periodic
one-way road. Speeds evolve with bounded random variation. The paper's six state
features are augmented with five twin estimates per mode: normalized energy,
service latency, service feasibility, link availability and remaining contact
budget. There are therefore 31 observation features per vehicle. The graph joins
vehicles within 230 m or sharing an RSU; Eq. 41 does not completely specify the
agent-level graph or aggregation architecture.

The digital twin is a simulator-generated **noisy estimate**, not a separately
trained forecaster or a connection to real vehicle hardware. Rewards, actual
service and energy use the true simulated quantities, not the noisy estimates.

Units and default physical assumptions:

| Quantity | Value / model |
| --- | --- |
| Slot and per-attempt service deadline | 1 second |
| Task size | 0.6 decimal MB, split into 3 equal packets |
| Arrival process | Poisson mean 2.3 packets per vehicle per slot |
| CPU intensity | 500 cycles/bit, heterogeneous by vehicle |
| Local / moving / parked / MEC CPU | 2.5 / 5 / 8 / 60 GHz, divided among reserved slots |
| Remote slots per RSU | 2 moving, 2 parked, 3 MEC |
| Transmission power, bandwidth | 0.1 W, 20 MHz |
| Rate | `B * log2(1 + SNR)` with stochastic fading and distance attenuation |
| Service time | CPU cycles / Hz + round-trip bit transfer / bit rate + mode RTT |
| Compute energy | Mode-specific calibrated J/Gcycle; local coefficient scales with CPU frequency squared |
| Communication energy | Transmission power × round-trip transfer time; cloud adds 0.4 J/Mbit backhaul cost |
| Twin prediction deviation | Multiplicative noise with 5% standard deviation |

These energy coefficients, fading, CPU partitions and capacities are explicit
calibrated assumptions; they are not measurements or constants recovered from
the paper. Contact duration is remaining RSU-region distance / vehicle speed,
which is a dimensionally consistent simplification of the mobility constraint.
Remote servers use reserved service slots, **not** the paper's full M/M/b queues.

**Metric boundary:** `deadline_success` measures whether an attempted packet's
computation/transmission service fits its one-second service and contact budget.
It does **not** include time spent waiting in the agent backlog and is not an
end-to-end application deadline guarantee. Failed attempts consume modeled energy
and stay queued for retry. Arrivals beyond current service capacity remain in the
queue. We report queue size, queue growth and energy per completed Mbit alongside
service success so a policy cannot appear efficient just by doing less work.
The finite 40-slot episodes cannot establish mathematical strong queue stability.

The paper reports 120 vehicles, 12 RSUs and 15 MB tasks, but its “1.5 GHz CPU
cycles per bit” mixes units. It does not supply sufficient constants, code, seeds
or exact GNN details to reproduce its figures uniquely. We use 500 **cycles/bit**
explicitly. No claim is made to match the paper's absolute latency/energy results.
Task caching, real traces, detailed packet radio interference, SUMO, and learned
digital-twin forecasting are not implemented.

## Training and evaluation protocol

The paper's 80/60 ReLU layers, batch size 32, replay size 1024, update interval 10
and Adam optimizer are retained. The demo uses learning rates 0.001, discount 0.95,
target rate 0.01 and Lyapunov coefficient 0.15. Its 0.001 target-rate symbol is
ambiguous in the extracted paper; `paper_training.json` records that interpretation
and learning rate 0.01 as a separate, unvalidated configuration.

The first 256 matching transitions initialize replay. Optional behavior cloning
adds 160 supervised actor updates before RL; this is an implementation extension,
and the `warmstart.pt` checkpoint is evaluated separately. During RL the actors
generate decisions with decaying Gaussian logit noise. Every ten environment
steps, **all agents** receive one critic and actor update. No teacher labels are
used in RL updates. The Lyapunov surrogate does use the replayed twin cost model.

Training seeds are 7, 19 and 41. Each has separate rollout seeds. Validation
traffic seeds 200000–200005 select `best.pt` by mean reward. Frozen checkpoints
are tested on traffic seeds 300000–300019, which are never used for gradient
updates or checkpoint selection. Comparators use the same exogenous trajectory
draws; their policy RNG is independent of the simulator RNG. Episodes are
time-limit truncations, so Bellman targets bootstrap across the boundary.

Reported 95% intervals use a normal approximation across 20 **traffic seeds**
for each trained model. They do not measure uncertainty over all possible
networks, simulator assumptions, or training runs. The three-training-seed
summary reports separate means and ranges; repeated traffic scenarios across
training seeds are not treated as 60 independent traffic samples.

