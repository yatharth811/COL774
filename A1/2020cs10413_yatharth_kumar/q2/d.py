import numpy as np
from StochasticGradientDescent import StochasticGradientDescent
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
model = StochasticGradientDescent(X=X, Y=Y, batch_size=sizes[0], learning_rate=0.001, epochs=1000000, threshold=thresholds[0])
print(model.descent())
data = model.gradientData

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [d[0] for d in data]
y = [d[1] for d in data]
z = [d[2] for d in data]

def update(frame):
  ax.clear()
  scatter = ax.scatter(x[:frame], y[:frame], z[:frame], marker='o')
  ax.set_xlabel('theta0')
  ax.set_ylabel('theta1')
  ax.set_zlabel('theta2')
  return scatter, 

ani = FuncAnimation(fig, update, frames=len(z), interval=0)
plt.show()