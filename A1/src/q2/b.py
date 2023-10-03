import numpy as np
from StochasticGradientDescent import StochasticGradientDescent
np.random.seed(0)
theta = np.array([3, 1, 2])
sigma = np.sqrt(2)
num_samples = 1000000

x1 = np.random.normal(3, 2, num_samples)
x2 = np.random.normal(-1, 2, num_samples)
noise = np.random.normal(0, sigma, num_samples)
X = np.column_stack((np.ones(num_samples), x1, x2))
Y = theta[0] + theta[1] * x1 + theta[2] * x2 + noise


sizes = [1, 100, 10000, 1000000]
thresholds = [1e-10, 1e-8, 1e-5, 1e-5]
for i, size in enumerate(sizes):
  model = StochasticGradientDescent(X=X, Y=Y, batch_size=size, learning_rate=0.001, epochs=1000000, threshold=thresholds[i])
  print(model.descent())
