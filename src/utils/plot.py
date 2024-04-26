import numpy as np
import matplotlib.pyplot as plt


N = 4

base = np.load('data/logs/biped/basic/eval_on_hardcore/evaluations.npz')
hardcore = np.load('data/logs/biped/hardcore_transfer/evaluations.npz')
combined = np.concatenate([base['results'],hardcore['results']])
x = np.concatenate([base['timesteps'],np.max(base['timesteps'])+hardcore['timesteps']])/6400
y_mean = np.convolve(np.mean(combined,axis=1),np.ones(N)/N,'same')
y_std = np.convolve(np.std(combined,axis=1),np.ones(N)/N,'same')
plt.plot(x,y_mean,label='Direct Transfer')
plt.fill_between(x,y_mean-y_std,y_mean+y_std, alpha=0.2)


hardcore = np.load('data/logs/biped/hardcore/evaluations.npz')
combined = hardcore['results']
x = hardcore['timesteps']/6400
y_mean = np.convolve(np.mean(combined,axis=1),np.ones(N)/N,'same')
y_std = np.convolve(np.std(combined,axis=1),np.ones(N)/N,'same')
plt.plot(x,y_mean,label='Hardcore Only')
plt.fill_between(x,y_mean-y_std,y_mean+y_std, alpha=0.2)


hardcore = np.load('data/logs/biped/trex/eval_on_hardcore/evaluations.npz')
combined = np.concatenate([base['results'],hardcore['results']])
x = np.concatenate([base['timesteps'],np.max(base['timesteps'])+hardcore['timesteps']])/6400
y_mean = np.convolve(np.mean(combined,axis=1),np.ones(N)/N,'same')
y_std = np.convolve(np.std(combined,axis=1),np.ones(N)/N,'same')
plt.plot(x,y_mean,label='TREX Transfer')
plt.fill_between(x,y_mean-y_std,y_mean+y_std, alpha=0.2)

plt.legend()
plt.show()
pass