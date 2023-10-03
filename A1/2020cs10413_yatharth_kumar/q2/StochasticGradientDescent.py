import numpy as np
import time

class StochasticGradientDescent:
  
  def __init__(self, X, Y, batch_size, learning_rate, epochs, theta =  np.array([0, 0, 0]), threshold = 1e-6):
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.X = X
    self.Y = Y
    self.m = len(X)
    self.epochs = epochs
    self.theta = theta
    self.threshold = threshold
    self.batches = self.m//batch_size
    self.k = min(100, self.batches)
    self.gradientData = []
    pass
  
  def J(self, NX, NY):
    sum = 0
    for i in range(self.batch_size):
      sum += (NX[i] @ self.theta - NY[i]) * (NX[i] @ self.theta - NY[i])
    return sum / (2 * self.m)
    
  def step(self, NX, NY):
    s = np.array([0.0, 0.0, 0.0])
    for i in range(self.batch_size):
        s[0] += (NX[i] @ self.theta - NY[i]) * NX[i][0]
        s[1] += (NX[i] @ self.theta - NY[i]) * NX[i][1]
        s[2] += (NX[i] @ self.theta - NY[i]) * NX[i][2]

    return s / self.batch_size
  
  def descent(self): 
    threshold = self.threshold
    pcost, ncost = 0, 0
    iterations = 0
    # start_time = int(time.time())
    for i in range(0, self.epochs):
      iterations += 1
      index = i % self.batches
      start = index * self.batch_size
      NX = self.X[start : start + self.batch_size]
      NY = self.Y[start : start + self.batch_size]
      self.theta = self.theta - self.step(NX, NY) * self.learning_rate
      self.gradientData.append(self.theta)
      ncost += self.J(NX, NY)
      if ((i + 1) % self.k == 0):
        ncost = ncost/self.k
        # print(ncost, pcost)
        if (abs(ncost - pcost) < threshold):
          break 
        pcost = ncost
        ncost = 0

      # print(self.theta, iterations)
    # print(int(time.time()) - start_time)
    return self.theta, iterations
