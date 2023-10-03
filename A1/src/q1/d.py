import numpy as np
import matplotlib.pyplot as plt
from utils import *
from LinearRegression import LinearRegression
from matplotlib.animation import FuncAnimation

# Read data from file
dir_x = "data/linearX.csv"
dir_y = "data/linearY.csv"
xs, ys = load_data(dir_x), load_data(dir_y)
xs = normalise(xs)

model = LinearRegression(theta=np.array([0,0]), alpha=0.01, m=100, xs=xs, ys=ys)
model.gradient_descent()
data = model.gradientData
x = [d[0][0] for d in data]
y = [d[0][1] for d in data]
z = [d[1] for d in data]

fig = plt.figure("Contour Plot")
def plot_loss(xs, ys):
  theta0_range = np.linspace(-3, 3, 100)
  theta1_range = np.linspace(-3, 3, 100)
  theta0_vals, theta1_vals = np.meshgrid(theta0_range, theta1_range)
  loss_vals = np.zeros_like(theta0_vals)
  for i in range(len(theta0_range)):
    for j in range(len(theta1_range)):
      theta0 = theta0_range[i]
      theta1 = theta1_range[j]
      predictions = theta0 + theta1 * xs
      loss_vals[j, i] = np.mean((predictions - ys)**2)/2
  
  plt.contour(theta0_vals, theta1_vals, loss_vals, cmap='viridis', levels=50)
  plt.xlabel('Theta 0')
  plt.ylabel('Theta 1')

def update(frame):
  fig.clear()
  plot_loss(xs, ys)
  scatter = plt.scatter(x[:frame], y[:frame], c='b', marker='o')
  return scatter, 

ani = FuncAnimation(fig, update, frames=len(z), interval=200)
plt.show()