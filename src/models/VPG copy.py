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
import metaworld

ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

training_envs = []
for name, env_cls in ml10.train_classes.items():
  env_no_render = env_cls()
  env_render = env_cls(render_mode='human')
  task = random.choice([task for task in ml10.train_tasks
                        if task.env_name == name])
  env_render.set_task(task)
  env_no_render.set_task(task)
  training_envs.append((env_render,env_no_render))

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

def rollout_policy(render,N,std,max_len=500, min_frames=0):
    D = []
    i = 0
    n_frames = 0
    while i < N or n_frames < min_frames:
        for env_num, (env_render, env_no_render) in enumerate(training_envs):
            if render:
                env = env_render
            else:
                env = env_no_render
            print(f'Rolling out policy {i+1}/{N}: {env_num}' + ' '*40,end='\r')
            # Set up next episode
            obs, _ = env.reset()
            done = False
            episode = []
            while True:
                n_frames += 1
                obs_tensor = torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float32)

                a_dist = Normal(actor(obs_tensor),std)
                a = a_dist.sample()
                logp = torch.sum(a_dist.log_prob(a))

                obs_new, reward, done, trunc, info = env.step(a.detach().numpy()[0])
                episode.append([obs,a,reward,logp,obs_new])
                obs = obs_new

                if render:
                    env.render()
                if done or trunc or len(episode) > max_len:
                    D.append(episode)
                    break
        
        render = False
        i += 1

    return D


if __name__ == '__main__':
    actor = Actor(39,4)
    critic = Critic(39)

    # actor.load_state_dict(torch.load('data/VPG/Third_attempt/actor_weights_499_-14.4783'))
    # critic.load_state_dict(torch.load('data/VPG/Third_attempt/critic_weights_499_-14.4783'))

    actor_optim = torch.optim.Adam(actor.parameters(),lr=0.001)
    critic_optim = torch.optim.Adam(critic.parameters(),lr=0.00025)

    lda = 0.97
    gamma = 0.98
    std = 2
    decay = 0.7
    decay_every = 100
    num_epiesodes = 1
    min_frames = 0

    try:
        best_best_score = -200
        for i in range(500):
            episodes = rollout_policy(False,num_epiesodes,std,500,min_frames)
            
            mat = []
            y = []
            for n_ep, episode in enumerate(episodes):
                print(f'Calculating Value {n_ep+1}/{len(episodes)}' + ' '*40,end='\r')
                v = 0
                for frame in episode[::-1]:
                    mat.append(frame[0])
                    v = v * gamma + frame[2]
                    y.append(v)
            mat = torch.from_numpy(np.array(mat,dtype='float32'))
            best_score = np.max(y)
            y = torch.from_numpy(np.array(y,dtype='float32')[:,None])
            critic_losses = []
            for val_epoch in range(20):
                print(f'Critic Update {val_epoch+1}/{20}' + ' '*40,end='\r')
                y_pred = critic(mat)
                v_loss = nn.functional.mse_loss(y_pred,y)
                critic_optim.zero_grad()
                v_loss.backward()
                critic_optim.step()
                critic_losses.append(v_loss.detach().item())
            
            actor_optim.zero_grad()
            scalar = len(episodes)
            actor_losses = []
            for n_ep, episode in enumerate(episodes):
                print(f'Actor Update {n_ep+1}/{len(episodes)}' + ' '*40,end='\r')
                cum_adv = 0
                obs = []
                obs_new = []
                r = []
                for frame in episode[::-1]:
                    obs.append(frame[0])
                    obs_new.append(frame[4])
                    r.append(frame[2])
                obs = torch.tensor(obs, dtype=torch.float32)
                obs_new = torch.tensor(obs_new, dtype=torch.float32)
                r = torch.tensor(r)
                delta = (gamma*critic(obs_new)+r - critic(obs))[0].detach()
                for n_frame, frame in enumerate(episode[::-1]):
                    cum_adv = delta[n_frame] + gamma * lda * cum_adv
                    agent_loss = (-frame[3]*cum_adv/scalar)
                    agent_loss.backward()
                    actor_losses.append(agent_loss.detach().item())
            actor_optim.step()

            testing = rollout_policy(False,1,0.1,500)
            v = 0
            y=[]
            for frame in testing[0][::-1]:
                v = v * gamma + frame[2]
                y.append(v)
            best_score = np.max(y)
            if best_score > best_best_score or i % 10 == 9:
                torch.save(actor.state_dict(), f'data/VPG/actor_weights_{i}_{np.round(best_score,4)}')
                torch.save(critic.state_dict(), f'data/VPG/critic_weights_{i}_{np.round(best_score,4)}')
                if best_score > best_best_score:
                    best_best_score = best_score
            if i % decay_every == decay_every-1 and std > 0.25:
                std *= decay
            print(f'Epoch: {i+1}, STD: {np.round(std,1)} Critic Loss: {np.round(np.mean(critic_losses),4)}, Agent Loss: {np.round(np.mean(actor_losses),4)}, Best Score: {np.round(best_score,4)}')



    except:
        pass
