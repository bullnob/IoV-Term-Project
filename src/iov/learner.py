from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import Config, EnvironmentConfig, TrainingConfig
from .environment import OffloadingEnv
from .matching import Matcher
from .networks import CentralCritic, GraphActor, soft_update


def tensors(states):
    return (torch.from_numpy(np.stack([s.observation for s in states])),
            torch.from_numpy(np.stack([s.adjacency for s in states])))


def bellman_target(rewards, terminated, next_values, gamma):
    return rewards + gamma * (1 - terminated) * next_values


class Learner:
    def __init__(self, config: Config):
        self.config = config
        c, t = config.environment, config.training
        torch.set_num_threads(t.threads)
        torch.manual_seed(t.seed)
        self.rng = np.random.default_rng(t.seed + 987)
        self.matcher = Matcher(c)
        self.actors = nn.ModuleList([GraphActor(OffloadingEnv.observation_size, c.subtasks, k,
                                              t.hidden, t.graph_hidden) for k in range(c.agents)])
        self.critics = nn.ModuleList([CentralCritic(c.agents, OffloadingEnv.observation_size,
                                                   c.subtasks, t.hidden) for _ in range(c.agents)])
        self.target_actors = deepcopy(self.actors).requires_grad_(False)
        self.target_critics = deepcopy(self.critics).requires_grad_(False)
        self.actor_optimizers = [torch.optim.Adam(a.parameters(), lr=t.actor_lr) for a in self.actors]
        self.critic_optimizers = [torch.optim.Adam(q.parameters(), lr=t.critic_lr) for q in self.critics]
        self.updates = 0

    def logits(self, obs, graph, target=False):
        actors = self.target_actors if target else self.actors
        return torch.stack([actor(obs, graph) for actor in actors], dim=1)

    def project(self, logits, states):
        values = logits.detach().cpu().numpy()
        modes = np.stack([self.matcher.solve(x, s.rsu, s.available) for x, s in zip(values, states, strict=True)])
        return F.one_hot(torch.from_numpy(modes), num_classes=5).float()

    @torch.no_grad()
    def act(self, state, noise=0.0):
        obs, graph = tensors([state])
        scores = self.logits(obs, graph)[0].numpy()
        if noise > 0:
            scores = scores + self.rng.normal(0, noise, scores.shape)
        return self.matcher.solve(scores, state.rsu, state.available)

    def imitate(self, transitions):
        """Optional, explicitly reported matching warm-start, not RL."""
        states = [x.state for x in transitions]
        obs, graph = tensors(states)
        labels = torch.from_numpy(np.stack([x.action for x in transitions])).long()
        losses = []
        for k, (actor, optimizer) in enumerate(zip(self.actors, self.actor_optimizers, strict=True)):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(actor(obs, graph).flatten(0, 1), labels[:, k].flatten())
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    def sync_targets(self):
        self.target_actors.load_state_dict(self.actors.state_dict())
        self.target_critics.load_state_dict(self.critics.state_dict())

    def update(self, transitions):
        t, c = self.config.training, self.config.environment
        states, next_states = [x.state for x in transitions], [x.next_state for x in transitions]
        obs, graph = tensors(states)
        next_obs, next_graph = tensors(next_states)
        actions = F.one_hot(torch.from_numpy(np.stack([x.action for x in transitions])).long(), 5).float()
        rewards = torch.from_numpy(np.stack([x.reward for x in transitions]))
        terminal = torch.tensor([x.terminated for x in transitions], dtype=torch.float32)
        costs = torch.from_numpy(np.stack([x.costs for x in states]))
        with torch.no_grad():
            next_actions = self.project(self.logits(next_obs, next_graph, target=True), next_states)
            targets = [bellman_target(rewards[:, k], terminal,
                                      q(next_obs, next_graph, next_actions), t.gamma)
                       for k, q in enumerate(self.target_critics)]
        critic_losses = []
        for k, (critic, optimizer) in enumerate(zip(self.critics, self.critic_optimizers, strict=True)):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(critic(obs, graph, actions), targets[k])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite critic loss")
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 5)
            optimizer.step()
            critic_losses.append(loss.item())

        # Hard feasible actions in the forward pass; softmax derivative in the
        # backward pass. The discrete projection is not exactly differentiable.
        with torch.no_grad():
            hard_joint = self.project(self.logits(obs, graph), states)
        actor_losses, gradient_norms = [], []
        for k, (actor, critic, optimizer) in enumerate(zip(self.actors, self.critics, self.actor_optimizers, strict=True)):
            optimizer.zero_grad(set_to_none=True)
            critic.requires_grad_(False)
            probabilities = actor(obs, graph).softmax(-1)
            own_action = probabilities + (hard_joint[:, k] - probabilities).detach()
            joint = torch.stack([own_action if j == k else hard_joint[:, j] for j in range(c.agents)], dim=1)
            drift = (probabilities * costs[:, k]).sum(-1).mean()
            loss = -critic(obs, graph, joint).mean() + t.lyapunov_phi * drift
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite actor loss")
            loss.backward()
            gradient_norms.append(float(nn.utils.clip_grad_norm_(actor.parameters(), 5)))
            optimizer.step()
            critic.requires_grad_(True)
            actor_losses.append(loss.item())
        soft_update(self.target_actors, self.actors, t.tau)
        soft_update(self.target_critics, self.critics, t.tau)
        self.updates += 1
        return {"critic_loss": float(np.mean(critic_losses)), "actor_loss": float(np.mean(actor_losses)),
                "actor_gradient_norm": float(np.mean(gradient_norms))}

    def save(self, path, metadata=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "config": asdict(self.config),
                    "actors": self.actors.state_dict(), "critics": self.critics.state_dict(),
                    "metadata": metadata or {}, "updates": self.updates}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        raw = data["config"]
        model = cls(Config(EnvironmentConfig(**raw["environment"]), TrainingConfig(**raw["training"])))
        model.actors.load_state_dict(data["actors"])
        model.critics.load_state_dict(data["critics"])
        model.sync_targets()
        model.updates = data["updates"]
        model.actors.eval()
        model.critics.eval()
        return model

