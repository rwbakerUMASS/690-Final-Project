import sys
sys.path.append('src/')

from stable_baselines3 import PPO
import gymnasium
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker, BipedalWalkerHardcore
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement, CallbackList
from typing import TYPE_CHECKING, List, Optional
import pickle
from tqdm import tqdm
from utils.helpers import Create_Random_Trajectories

class TrajectoryRanking():

    def __init__(self, policies, env, num_per_policy=10):
        self.model = nn.Sequential(
            nn.Linear(24,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Tanh()
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

    def _make_pairs(self, auto=False, test=False, random=False):
        pairs = []
        prefs = []
        num_per_policy = self.num_per_policy
        if test:
            num_per_policy = int(num_per_policy * 0.2)
        for policy in self.policies:
            for i in tqdm(range(num_per_policy)):
                if random:
                    pair, rews = Create_Random_Trajectories(2, 500)
                else:
                    pair, rews = rollout_policy(policy, self.env, 2, 500, i+int(test)*num_per_policy)
                length = np.min([len(t) for t in pair])
                pair = [t[:length] for t in pair]
                pairs.append(pair)
                # print(f'{i}:{np.round(rews[0],1)} vs {np.round(rews[1],1)}')
                pref = None
                if auto:
                    pref = np.argmax(rews)
                else:
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

    def train(self, num_iter,load=False,random=False):
        if load:
            pairs = pickle.load(open('data/trex/pairs','rb'))
            prefs = pickle.load(open('data/trex/prefs','rb'))
            test  = pickle.load(open('data/trex/test','rb'))
        else:
            pairs, prefs = self._make_pairs(True,random=random)
            test = self._make_pairs(True,True,random=random)
            pickle.dump(pairs,open('data/trex/pairs','wb'))
            pickle.dump(prefs,open('data/trex/prefs','wb'))
            pickle.dump(test,open('data/trex/test','wb'))
        loss_criterion = nn.CrossEntropyLoss()
        best_acc = 0
        best_model = None
        for i in range(num_iter):
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
                self.optim.step()
            acc = []
            with torch.no_grad():
                for pair in range(len(test[0])):
                    t_i, t_j = test[0][pair]
                    preference = torch.tensor(test[1][pair])
                    t_i = torch.tensor(t_i)
                    t_j = torch.tensor(t_j)
                    r_i = torch.sum(self.model(t_i)).unsqueeze(-1)
                    r_j = torch.sum(self.model(t_j)).unsqueeze(-1)
                    cum_r = torch.cat((r_i,r_j))
                    loss = loss_criterion(cum_r,preference)
                    acc.append(int(torch.argmax(cum_r).detach().item()==preference.detach().item()))
            print(f'iter:{i}: {np.round(np.mean(acc),3)}')
            if np.mean(acc) > best_acc:
                print('NEW BEST MODEL!')
                best_acc = np.mean(acc)
                best_model = self.model.state_dict()
        
        self.model.load_state_dict(best_model)


def rollout_policy(policy, env, n_eps=1, max_steps=1600, seed=0, render=False):
    demos = []
    rewards = []
    for i in range(n_eps):
        done = False
        truncated = False
        steps = 0
        traj = []
        obs, _ = env.reset(seed=seed)
        cum_rew = 0
        while not done and not truncated and steps < max_steps:
            traj.append(obs)
            action, _states = policy.predict(obs,deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            # env.render()
            cum_rew += reward
            steps += 1
        traj.append(obs)
        rewards.append(cum_rew)
        demos.append(np.array(traj))
    return demos, rewards

class TrexEnv(BipedalWalker):
    def __init__(self, model, max_steps=2000, render_mode: Optional[str] = None, hardcore: bool = False):
        super().__init__(render_mode, hardcore)
        self.model = model
        self.max_steps = max_steps
        self.steps = 0

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        rew = rew * self.model(torch.tensor(obs)).detach().item()
        if self.steps > self.max_steps:
            term = True
        return obs, rew, term, trunc, info 

    def reset(self, seed=None, options=None):
        self.steps = 0
        return super().reset(seed=seed,options=options)
    
class EvalSetSeed(BipedalWalker):
    '''
        This BipedalWalker env locks the seed to ensure consistency when evaluating agent performance
        ## Params
            - max_steps: maximum number of frames allowed
            - render_mode: Sets env render mode
            - hardcore: Sets whether hardcore mode is enabled
    '''
    def __init__(self, max_steps, render_mode: Optional[str] = None, hardcore: bool = False):
        super().__init__(render_mode, hardcore)
        self.max_steps = max_steps
        self.steps = 0

    def step(self, action):
        self.steps += 1
        obs, rew, term, trunc, info = super().step(action)
        if self.steps > self.max_steps:
            term = True
        return obs, rew, term, trunc, info

    def reset(self, seed=None, options=None):
        self.steps = 0
        return super().reset(seed=0)


if __name__ == '__main__':
    policy = PPO.load("data\\logs\\biped\\basic\\best_model")
    env = gymnasium.make('BipedalWalker-v3',render_mode=None,hardcore=True)
    trex = TrajectoryRanking([policy],env,6000)
    trex.train(20,True)
    torch.save(trex.model.state_dict(),'trex_reward_model_NEW')
    trex.model.load_state_dict(torch.load('trex_reward_model_NEW'))

    trex_env_spec = lambda: TrexEnv(model=trex.model,render_mode=None,hardcore=True)
    env_spec2 = lambda: EvalSetSeed(2000, render_mode=None,hardcore=True)
    # rollout_policy(policy,gymnasium.make('BipedalWalkerHardcore-v3',render_mode='human'),1,500,0,True)
    vec_env = make_vec_env(trex_env_spec, n_envs=4)
    # model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=1600, batch_size=64, gae_lambda=0.95, gamma=0.999, n_epochs=10, ent_coef=0.0, learning_rate=3e-4, clip_range=0.18) # model for task #1
    model = policy
    model.env = vec_env

    no_progress_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=100, verbose=1)
    score_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)

    eval_callback_hardcore = EvalCallback(env_spec2(),
                                    callback_on_new_best=score_threshold_callback,
                                    # callback_after_eval=no_progress_callback,
                                    best_model_save_path="./data/logs/biped/trex/2",
                                    log_path="./data/logs/biped/trex/2",
                                    eval_freq=1600,
                                    deterministic=True,
                                    render=False)

    eval_callback_trex = EvalCallback(trex_env_spec(),
                                    eval_freq=1600,
                                    deterministic=True,
                                    render=False)

    callback_list = CallbackList([eval_callback_hardcore])

    model.learn(total_timesteps=1e7,callback=callback_list)
pass