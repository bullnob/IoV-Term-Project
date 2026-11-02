# How to understand and present the project

You can present the delivered model without training anything. Start by opening
`artifacts/demo/showcase.html`. It contains real recorded results from the saved
model and needs no Internet connection. Use the live Python dashboard when you
want to generate new traffic scenarios or change workload.

## The idea in ordinary language

Imagine several cars generating computing jobs. Each car can compute a small
part of a job itself, ask a moving vehicle, ask a parked vehicle, send it to an
RSU edge server, or send it to a cloud. The cheap option is not always available
or fast enough, and multiple cars can compete for the same processor.

The **digital twin** is the simulator's estimated snapshot of those resources.
The **graph** tells each car which neighbors it can coordinate with. The **actor**
is a neural network that proposes where to send work. The **critic** learns how
valuable that choice is by observing later rewards. Each car has its own actor
and critic. Matching ensures the proposed choices fit shared resource capacities.

Reinforcement learning learns from outcomes: observe → decide → simulate → receive
reward → remember the transition → update the networks. It is different from
ordinary supervised learning, where every example comes with a correct answer.
We do use a short supervised matching warm-start first, clearly measured as a
separate stage. The rest of training updates the networks from rewards.

The **Lyapunov term** discourages growing queues while accounting for energy.
The practical tradeoff is: saving power is not useful if unfinished work piles up.
That is why the dashboard includes queue measurements as well as energy.

## A five-minute demonstration

1. **0:00–0:40 — State the problem.** “Vehicles generate computing tasks. We want
   to share that work efficiently while respecting finite server capacity and
   a service-time budget.” Point to the five mode colors.
2. **0:40–1:20 — Show the untrained policy.** Choose “Untrained neural network”
   and press Play. Explain that the lines show assigned resources and the small
   bars show three reserved task slots for each vehicle.
3. **1:20–2:10 — Switch to the trained policy.** Keep the same traffic seed. Compare
   reward, energy, service success and queues. These are paired scenarios, not
   different cherry-picked road conditions. Use the measured numbers in RESULTS.md.
4. **2:10–3:00 — Explain the learning chart.** Gray is exploratory training;
   green is noise-free validation. Show untrained → matching warm-start → trained
   scores. State that matching imitation and reinforcement learning are separate.
5. **3:00–4:00 — Show the held-out table.** Explain that these 20 traffic seeds
   were not used for weight updates or checkpoint selection. The matching planner
   is a useful model-based reference; the neural policy need not beat that planner.
6. **4:00–5:00 — Discuss scope.** “This is an implementation of the paper's
   training idea in an explicit synthetic environment. It is not an exact numerical
   reproduction of the article, a real vehicle deployment, or proof of queue
   stability.” Describe what you would add next: realistic traces, full queueing
   delays, more vehicles, and graph/no-graph ablations.

In the **live** dashboard, try arrival load 0.5× and 2×. At high load, queues can
grow even when service success looks good: arrivals can exceed available service.
The static benchmark table remains the standard-load benchmark; only the top
cards and animation change with the live controls.

## The slides to make

Use five slides: (1) the offloading problem; (2) the observe/actor/matching/critic
pipeline; (3) the paper-to-code mapping; (4) `artifacts/demo/results.png` and the
three-seed summary; (5) limitations and future work. Cite the paper on the method
slide. Do not copy its published graphs and label them as your experimental results.

Useful truthful phrasing:

> “I implemented a graph-assisted multiagent actor–critic offloading prototype
> based on Algorithm 2, with a capacity-constrained matching planner for initial
> experience generation and action feasibility. I evaluated the saved model on
> held-out synthetic traffic and documented my simulator assumptions.”

If asked about assistance, explain that implementation and experiment execution
were AI-assisted, and that you can explain the method, code, and evidence. Follow
your course's attribution rules.

## Questions you are likely to get

**Why not always use the matching solver?** It is a strong immediate model-based
planner when the estimated costs are available. RL can learn longer-term effects
and amortize decision preferences, but our actor still needs a matching projection.
We have not established a runtime advantage over the matching baseline.

**Why use multiple agents?** Every vehicle has its own state and policy; a decision
affects others through shared resource capacity. Centralized critics can use the
joint situation during training, while actors aggregate local neighbor information.

**What does the critic output?** An estimate of discounted future reward, not a
probability or a directly measured physical energy value.

**What are target networks and replay?** Replay reuses past transitions and reduces
temporal correlation. Target networks change slowly, making critic regression targets
less unstable while both policy and value estimates learn.

**Is matching data sufficient by itself?** No. The simulator must execute assignments
to produce rewards and next states. Those full transitions train the critic. Static
matching labels alone would train an imitation model, not this RL system.

**Does 99% service success mean every task finishes within one second of arrival?**
No. The simulator's service metric excludes backlog waiting time. Always state
“within the modeled per-attempt service deadline” and show queue size alongside it.

**Did it reproduce the research paper?** It implements the central training components
with explicit engineering choices. The paper lacks enough detail for an exact
reproduction, and the delivered demo uses a smaller synthetic network.

**How can I prove the neural network is actually used?** Switch between trained
and untrained checkpoints on the same seed. Inspect `learner.act`: actor logits go
to the matching projection. Only the planner baseline uses direct twin costs.
Tests also verify that every agent's actor and critic parameters change during RL.

## Retrain and inspect a model

Follow the commands in README.md, using a new output directory. The default run
is CPU-sized and its measured runtime is recorded in `training_manifest.json`;
other machines can be slower. Keep a backup of the supplied checkpoints before
experimenting. First change only one setting, such as arrival rate or twin noise,
so you can explain its effect. Changing the number of vehicles requires retraining.

You can inspect `matching_data.npz` with `numpy.load(..., allow_pickle=False)`,
read `history.csv` in a spreadsheet, and use the SVG chart in slides without
pixelation. `best.pt` is chosen by validation reward, while `final.pt` is simply
the last training iterate; later training is not automatically better.

