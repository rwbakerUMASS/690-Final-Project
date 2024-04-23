import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


vec_env = make_vec_env("BipedalWalker-v3", n_envs=4)    #task #1
vec_env2 = make_vec_env("BipedalWalkerHardcore-v3", n_envs=4)   # task #2

model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=1600, batch_size=64, gae_lambda=0.95, gamma=0.999, n_epochs=10, ent_coef=0.0, learning_rate=3e-4, clip_range=0.18) # model for task #1
model2 = PPO("MlpPolicy", vec_env2, verbose=1, n_steps=2048, batch_size=64, gae_lambda=0.95, gamma=0.99, n_epochs=10, ent_coef=0.001, learning_rate=2.5e-4, clip_range=0.20)    # model for task #2

# train and save model. (total_timesteps = n_envs * n_steps * iterations)
model.learn(total_timesteps=640000)
model.save("ppo_bipedalwalker")

# model = PPO.load("ppo_cartpole", weights_only=False)

# test runs on env1
obs = vec_env.reset()
total_reward = [0,0,0,0]
num_runs = 10
while num_runs > 0:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    total_reward += rewards
    vec_env.render("human")
    for i in range(len(dones)):
        if dones[i]:
            print('window: ', i, 'reward: ', total_reward[i])
            total_reward[i] = 0
            num_runs -= 1
vec_env.close()

# transferring policy from env1 (task #1) to env2 (task #2)
policy1 = model.policy
model2.policy = policy1

# train on task #2, with policy from task #1. (total_timesteps = n_envs * n_steps * iterations)
model2.learn(total_timesteps=3276800)

# testing of task #2
obs2 = vec_env2.reset()
total_reward2 = [0,0,0,0]
num_runs2 = 20
while num_runs2 > 0:
    action2, _states2 = model2.predict(obs)
    obs2, rewards2, dones2, info2 = vec_env2.step(action2)
    total_reward2 += rewards2
    vec_env2.render("human")
    for i in range(len(dones2)):
        if dones2[i]:
            print('window: ', i, 'reward: ', total_reward2[i])
            total_reward2[i] = 0
            num_runs2 -= 1
    vec_env2.close()