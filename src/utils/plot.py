import numpy as np
import matplotlib.pyplot as plt


N = 5
N2 = 1

base = np.load('data/logs/biped/basic/eval_on_hardcore/evaluations.npz')
hardcore = np.load('data/logs/biped/hardcore_transfer/evaluations.npz')
combined = np.concatenate([base['results'],hardcore['results']])[::N2]
y_mean = np.array([])
y_std = np.array([])
for i in range(len(combined)-N):
    y_mean = np.append(y_mean,np.mean(combined[i:i+N]))
    y_std = np.append(y_std,np.std(combined[i:i+N]))
x = np.concatenate([base['timesteps'],np.max(base['timesteps'])+hardcore['timesteps']])[::N2][:len(y_mean)]/6400
plt.plot(x,y_mean,label='Direct Transfer')
plt.fill_between(x,y_mean-y_std,y_mean+y_std, alpha=0.2)


hardcore = np.load('data/logs/biped/hardcore/evaluations.npz')
combined = hardcore['results'][::N2]
y_mean = np.array([])
y_std = np.array([])
for i in range(len(combined)-N):
    y_mean = np.append(y_mean,np.mean(combined[i:i+N]))
    y_std = np.append(y_std,np.std(combined[i:i+N]))

x = hardcore['timesteps'][::N2][:len(y_mean)]/(6400)
plt.plot(x,y_mean,label='Hardcore Only')
plt.fill_between(x,y_mean-y_std,y_mean+y_std, alpha=0.2)


hardcore = np.load('data/logs/biped/trex/2/evaluations.npz')
combined = np.concatenate([base['results'],hardcore['results']])[::N2]
y_mean = np.array([])
y_std = np.array([])
for i in range(len(combined)-N):
    y_mean = np.append(y_mean,np.mean(combined[i:i+N]))
    y_std = np.append(y_std,np.std(combined[i:i+N]))
x = np.concatenate([base['timesteps'],np.max(base['timesteps'])+hardcore['timesteps']])[::N2][:len(y_mean)]/6400
plt.plot(x,y_mean,label='TREX Transfer')
plt.fill_between(x,y_mean-y_std,y_mean+y_std, alpha=0.2)

plt.legend()
plt.show()
pass