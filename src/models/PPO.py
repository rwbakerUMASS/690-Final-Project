from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# # Parallel environments
# vec_env = make_vec_env(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['door-open-v2-goal-observable'], n_envs=16)
env_spec = lambda: gymnasium.make('BipedalWalker-v3',hardcore=True)
vec_env = make_vec_env(env_spec, n_envs=25)

model = PPO("MlpPolicy", vec_env, verbose=1,n_steps=1600)
# model = PPO.load("ppo_cartpole")
# model.env = vec_env

eval_callback = EvalCallback(env_spec(), best_model_save_path="./data/logs/biped/basic/",
                             log_path="./data/logs/biped/basic/", eval_freq=500,
                             deterministic=True, render=False)

model.learn(total_timesteps=1000000,progress_bar=True,callback=eval_callback)
model.save("biped")

# del model # remove to demonstrate saving and loading

model = PPO.load("biped")

# obs = vec_env.reset()
env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['door-open-v2-goal-observable']
env = env()
env.render_mode = 'human'
env = gymnasium.make('BipedalWalker-v3',render_mode="human",hardcore=False)
obs, _ = env.reset()

while True:
    action, _states = model.predict(obs,deterministic=True)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render()