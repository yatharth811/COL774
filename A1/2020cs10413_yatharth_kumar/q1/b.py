import matplotlib.pyplot as plt
import numpy as np
from utils import *
from LinearRegression import LinearRegression
  
dir_x = "data/linearX.csv"
dir_y = "data/linearY.csv"
xs, ys = load_data(dir_x), load_data(dir_y)
xs = normalise(xs)

model = LinearRegression(theta=np.array([0,0]), alpha=0.01, m=100, xs=xs, ys=ys)
model.gradient_descent()

def plot_hypothesis(xs, ys, theta):
  plt.scatter(xs, ys)
  x = np.linspace(-2, 5)
  plt.plot(x, theta[0] + theta[1] * x, '-')
  plt.xlabel('acidity of wine')
  plt.ylabel('density of wine')
  plt.show()

plot_hypothesis(xs, ys, model.theta)