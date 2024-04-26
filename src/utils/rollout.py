import sys
sys.path.append('src/')

from utils.traj_ranking import rollout_policy
import gymnasium
from stable_baselines3 import PPO

models = {
    'base': 'data/logs/biped/basic/best_model',
    'hardcore': 'data/logs/biped/hardcore/best_model',
    'hardcore_transfer': 'data/logs/biped/hardcore_transfer/best_model',
    'trex': 'data/logs/biped/trex/best_model'
}


env = gymnasium.make('BipedalWalkerHardcore-v3',render_mode='human')
for policy in models.keys():
    model = PPO.load(models[policy])
    rollout_policy(model, env, 1,2000,0)