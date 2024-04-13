import numpy as np
import random
import gymnasium
import math
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.distributions.normal import Normal
import cv2
import copy

env_render = gymnasium.make('BipedalWalker-v3',render_mode="human",hardcore=False)
env_no_render = gymnasium.make('BipedalWalker-v3',hardcore=False)

class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        super().__init__()
        self.n_actions = n_actions
        self.mu = nn.Sequential(
            nn.Linear(state_dim, 300),
            activation(),
            nn.Linear(300, 300),
            activation(),
            nn.Linear(300, n_actions)
        )
    
    def forward(self, X):
        return self.mu(X)

class Critic(nn.Module):
    def __init__(self, state_dim, activation=nn.Tanh):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 300),
            activation(),
            nn.Linear(300, 300),
            activation(),
            nn.Linear(300, 1)
        )
    
    def forward(self, X):
        return self.layers(X)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def get_policy(obs):
    mu, std = actor(obs)
    return Normal(mu,std)

# make action selection function (outputs int actions, sampled from policy)
def get_action(obs):
    return get_policy(obs).sample().detach().numpy()[0]

# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights.unsqueeze(1)).mean()

def rollout_policy(render,N,std,max_len = 500):
    D = []
    for i in range(N):
        print(f'Rolling out policy {i+1}/{N}   ',end='\r')
        if render:
            env = env_render
        else:
            env = env_no_render

        # Set up next episode
        obs, _ = env.reset()
        done = False
        episode = []
        while True:
            obs_tensor = torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float32)

            a_dist = Normal(actor(obs_tensor),std)
            a = a_dist.sample()
            logp = torch.sum(a_dist.log_prob(a))

            obs_new, reward, done, trunc, info = env.step(a.detach().numpy()[0])
            episode.append([obs,a,reward,logp,obs_new])
            obs = obs_new

            if render:
                env.render()
            if done or len(episode) > max_len:
                render = False
                D.append(episode)
                break

    return D



actor = Actor(24,4)
actor_optim = torch.optim.Adam(actor.parameters(),lr=0.001)

critic = Critic(24)
critic_optim = torch.optim.Adam(critic.parameters(),lr=0.00025)

lda = 0.97
gamma = 0.98

try:
    for i in range(500):
        episodes = rollout_policy(False,10,2,5000)
        
        mat = []
        y = []
        for episode in episodes:
            v = 0
            for frame in episode[::-1]:
                mat.append(frame[0])
                v = v * gamma + frame[2]
                y.append(v)
        mat = torch.from_numpy(np.array(mat,dtype='float32'))
        best_score = np.max(y)
        y = torch.from_numpy(np.array(y,dtype='float32')[:,None])
        for _ in range(5):
            y_pred = critic(mat)
            v_loss = nn.functional.mse_loss(y_pred,y)
            critic_optim.zero_grad()
            v_loss.backward()
            critic_optim.step()
        
        actor_optim.zero_grad()
        scalar = len(episodes)
        for episode in episodes:
            delta_cum = 0
            for frame in episode[::-1]:
                obs,a,r,logp, obs_new = frame
                obs = torch.tensor(obs, dtype=torch.float32)
                obs_new = torch.tensor(obs_new, dtype=torch.float32)
                delta = (gamma*critic(obs_new)+r - critic(obs))[0].detach()
                delta_cum = delta + gamma * lda * delta_cum
                agent_loss = (-logp*delta_cum/scalar)
                agent_loss.backward()
        actor_optim.step()

        testing = rollout_policy(True,1,0.01,500)
        v = 0
        y=[]
        for frame in testing[0][::-1]:
            v = v * gamma + frame[2]
            y.append(v)
        best_score = np.max(y)
        print(f'Epoch: {i+1}, Critic Loss: {np.round(v_loss.detach().item(),4)}, Agent Loss: {np.round(agent_loss.detach().item(),4)}, Best Score: {np.round(best_score,4)}')



except:
    pass
