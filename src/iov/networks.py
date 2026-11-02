"""Graph actors and separate centralized critics, one pair per vehicle."""
import torch
from torch import nn


class GraphActor(nn.Module):
    def __init__(self, observation_size, subtasks, agent, hidden=(80, 60), graph_hidden=32):
        super().__init__()
        self.agent = agent
        self.subtasks = subtasks
        self.node_encoder = nn.Linear(observation_size, graph_hidden)
        self.policy = nn.Sequential(nn.Linear(2 * graph_hidden, hidden[0]), nn.ReLU(),
                                    nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                                    nn.Linear(hidden[1], subtasks * 5))
        nn.init.uniform_(self.policy[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.policy[-1].bias)

    def forward(self, observations, adjacency):
        encoded = torch.relu(self.node_encoder(observations))
        neighbors = adjacency[:, self.agent]
        neighbors = neighbors / neighbors.sum(-1, keepdim=True).clamp_min(1)
        context = (encoded * neighbors.unsqueeze(-1)).sum(1)
        features = torch.cat([encoded[:, self.agent], context], dim=-1)
        return self.policy(features).reshape(-1, self.subtasks, 5)


class CentralCritic(nn.Module):
    def __init__(self, agents, observation_size, subtasks, hidden=(80, 60)):
        super().__init__()
        width = agents * observation_size + agents * agents + agents * subtasks * 5
        self.value = nn.Sequential(nn.Linear(width, hidden[0]), nn.ReLU(),
                                   nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                                   nn.Linear(hidden[1], 1))

    def forward(self, observations, adjacency, actions):
        inputs = torch.cat([observations.flatten(1), adjacency.flatten(1), actions.flatten(1)], dim=1)
        return self.value(inputs).squeeze(-1)


@torch.no_grad()
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters(), strict=True):
        target_param.lerp_(param, tau)

