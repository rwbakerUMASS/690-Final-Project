from VPG import Actor, Critic, rollout_policy, env_no_render, env_render
import torch
from torch.distributions.normal import Normal
import numpy as np

actor = Actor(24,4)
actor_optim = torch.optim.Adam(actor.parameters(),lr=0.001)

critic = Critic(24)
critic_optim = torch.optim.Adam(critic.parameters(),lr=0.00025)


actor.load_state_dict(torch.load('data/VPG/Second_Attempt/actor_weights_9_5.165'))
critic.load_state_dict(torch.load('data/VPG/Second_Attempt/critic_weights_9_5.165'))

def rollout_policy(render,N,std,max_len=500):
    D = []
    for i in range(N):
        print(f'Rolling out policy {i+1}/{N}' + ' '*40,end='\r')
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

rollout_policy(True,1,2,500)