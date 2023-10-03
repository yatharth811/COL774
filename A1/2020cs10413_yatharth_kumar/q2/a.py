import numpy as np

np.random.seed(0)
theta = np.array([3, 1, 2])
sigma = np.sqrt(2)
num_samples = 1000000

x1 = np.random.normal(3, 2, num_samples)
x2 = np.random.normal(-1, 2, num_samples)
noise = np.random.normal(0, sigma, num_samples)
y = theta[0] + theta[1] * x1 + theta[2] * x2 + noise
data = np.column_stack((np.ones(num_samples), x1, x2, noise, y))
np.savetxt('dataset.csv', data, delimiter=',')