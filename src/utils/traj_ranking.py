from stable_baselines3 import PPO
import gymnasium
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class TrajectoryRanking():

    def __init__(self, policies, env, num_per_policy=10):
        self.model = nn.Sequential(
            nn.Linear(24,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

        self.optim = torch.optim.Adam(self.model.parameters())

        if not isinstance(policies, list):
            policies=[policies]

        self.policies = policies
        self.env = env
        self.num_per_policy = num_per_policy

    def predict_return(self, traj):
        if not isinstance(traj, torch.Tensor):
            traj = torch.tensor(traj)
        return torch.sum(self.model(traj))

    def _make_pairs(self):
        pairs = []
        prefs = []
        for policy in self.policies:
            for i in range(self.num_per_policy):
                pairs.append(rollout_policy(policy, self.env, 2, 500, np.random.randint(0,100)))
                pref = None
                while pref is None:
                    pref = input('Enter Preference ("0" for first, "1" for second):')
                    try:
                        pref = int(pref)
                        if pref != 0 and pref != 1:
                            pref = None
                    except:
                        pref = None
                prefs.append(pref)
        return pairs, prefs

    def train(self, num_iter):
        pairs, prefs = self._make_pairs()
        loss_criterion = nn.CrossEntropyLoss()
        for i in range(num_iter):
            losses = []
            for pair in range(len(pairs)):
                self.optim.zero_grad()
                t_i, t_j = pairs[pair]
                preference = torch.tensor(prefs[pair])
                t_i = torch.tensor(t_i)
                t_j = torch.tensor(t_j)
                r_i = torch.sum(self.model(t_i)).unsqueeze(-1)
                r_j = torch.sum(self.model(t_j)).unsqueeze(-1)
                cum_r = torch.cat((r_i,r_j))
                loss = loss_criterion(cum_r,preference)
                loss.backward()
                losses.append(loss.detach().item())
                self.optim.step()
            print(f'iter:{i}: {np.round(np.mean(losses),3)}')


def rollout_policy(policy, env, n_eps=1, max_steps=1600, seed=0, render=False):
    demos = []
    for i in range(n_eps):
        done = False
        truncated = False
        steps = 0
        traj = []
        obs, _ = env.reset(seed=seed)
        while not done and not truncated and steps < max_steps:
            traj.append(obs)
            action, _states = policy.predict(obs,deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            steps += 1
        traj.append(obs)
        demos.append(np.array(traj))
    return demos


policy = PPO.load("ppo_cartpole")
env = gymnasium.make('BipedalWalker-v3',render_mode="human",hardcore=True)
trex = TrajectoryRanking([policy],env,4)
trex.train(100)
pass