import sys
sys.path.append('src/')

# from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement, CallbackList
from utils.traj_ranking import rollout_policy, EvalSetSeed
import numpy as np

# # Parallel environments
# vec_env = make_vec_env(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['door-open-v2-goal-observable'], n_envs=16)
env_spec = lambda: gymnasium.make('BipedalWalker-v3',hardcore=False)
env_spec2 = lambda: gymnasium.make('BipedalWalker-v3',hardcore=True)
hardcore_eval = lambda: EvalSetSeed(2000, render_mode=None,hardcore=True)
vec_env = make_vec_env(env_spec, n_envs=4)
vec_env2 = make_vec_env(env_spec2, n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=1600, batch_size=64, gae_lambda=0.95, gamma=0.999, n_epochs=10, ent_coef=0.0, learning_rate=3e-4, clip_range=0.18) # model for task #1
model2 = PPO("MlpPolicy", vec_env2, verbose=1, n_steps=2048, batch_size=64, gae_lambda=0.95, gamma=0.99, n_epochs=10, ent_coef=0.001, learning_rate=2.5e-4, clip_range=0.20)    # model for task #2

# model = PPO.load("ppo_cartpole")
# model.env = vec_env

no_progress_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=10, verbose=1)
score_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)

eval_callback_base = EvalCallback(env_spec(),
                                  callback_after_eval=no_progress_callback,
                                  callback_on_new_best=score_threshold_callback,
                                  best_model_save_path="./data/logs/biped/basic/",
                                  log_path="./data/logs/biped/basic/eval_on_base",
                                  eval_freq=1600,
                                  deterministic=False,
                                  render=False)

eval_callback_hardcore = EvalCallback(env_spec2(),
                                      log_path="./data/logs/biped/basic/eval_on_hardcore",
                                      eval_freq=1600,
                                      deterministic=False,
                                      render=False)

callback_list = CallbackList([eval_callback_base,eval_callback_hardcore])

# model.learn(total_timesteps=1e10,callback=callback_list)
# model.save("biped")

# del model # remove to demonstrate saving and loading

model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=1600, batch_size=64, gae_lambda=0.95, gamma=0.999, n_epochs=10, ent_coef=0.0, learning_rate=3e-4, clip_range=0.18) # model for task #1
# model = PPO.load("biped")
model.env = vec_env2

eval_callback_hardcore2 = EvalCallback(hardcore_eval(),
                                #   callback_after_eval=no_progress_callback,
                                  callback_on_new_best=score_threshold_callback,
                                  best_model_save_path="data/logs/biped/hardcore",
                                  log_path="data/logs/biped/hardcore",
                                  eval_freq=1600,
                                  deterministic=False,
                                  render=False)

model.learn(total_timesteps=1e7,progress_bar=True,callback=eval_callback_hardcore2)

# obs = vec_env.reset()
# env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['door-open-v2-goal-observable']
# env = env()
# env.render_mode = 'human'
env = gymnasium.make('BipedalWalker-v3',render_mode="human",hardcore=True)

rollout_policy(model,env,seed=np.random.randint(0,100))