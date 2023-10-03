import numpy as np

class LogisticRegression:
  def __init__(self, xs, ys, zs, theta):
    self.xs = xs
    self.ys = ys
    self.zs = zs
    self.theta = theta
    self.m = len(xs)
    self.threshold = 1e-12
  
  def h(self, x):
    z = self.theta @ np.array(x)
    return 1.0/(1 + np.exp(-z))
  
  def logLikelihood(self):
    sum = 0.0
    for i in range(0, self.m):
      x, y = [1, self.xs[i], self.ys[i]], self.zs[i]
      val = self.h(x)
      sum += (y * np.log(val) + (1 - y) * np.log(1 - val))      
    return sum

  def step(self):
    s1, s2, s3 = 0, 0, 0
    for i in range(self.m):
      x, y = [1, self.xs[i], self.ys[i]], self.zs[i]
      val = self.h(x)
      s1 += (y - val)
      s2 += (y - val) * self.xs[i]
      s3 += (y - val) * self.ys[i]
    return np.array([s1, s2, s3])
  
  def hessian(self):
    H = np.zeros((3,3))
    for k in range(3):
      for j in range(3):
        for i in range(self.m):
            x = [1, self.xs[i], self.ys[i]]
            val = self.h(x)
            H[k][j] += x[j] * x[k] * val * (val - 1)
    return np.linalg.inv(H)
  
  def newton(self):
    threshold = self.threshold
    while True:
      prev = self.logLikelihood()
      self.theta = self.theta - self.hessian() @ self.step()
      if (abs(self.logLikelihood() - prev) < threshold):
        break
      
    return self.theta