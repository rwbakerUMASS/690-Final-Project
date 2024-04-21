import numpy as np
import matplotlib.pyplot as plt

f = np.load('./data/logs/evaluations.npz')
print(f.files)
x = f['timesteps']
N = 10
y_mean = np.convolve(np.mean(f['results'],axis=1),np.ones(N)/N,'same')
y_min = np.convolve(np.min(f['results'],axis=1),np.ones(N)/N,'same')
y_max = np.convolve(np.max(f['results'],axis=1),np.ones(N)/N,'same')
plt.plot(x,y_mean)
# plt.plot(x,y_min)
plt.plot(x,y_max)
plt.show()
pass