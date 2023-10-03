import numpy as np
from utils import *
from LinearRegression import LinearRegression

# Read data from file
dir_x = "data/linearX.csv"
dir_y = "data/linearY.csv"
xs, ys = load_data(dir_x), load_data(dir_y)
xs = normalise(xs)

# Do Gradient Descent
model = LinearRegression(theta=np.array([0,0]), alpha=0.01, m=100, xs=xs, ys=ys)
model.gradient_descent()
print(model.theta)