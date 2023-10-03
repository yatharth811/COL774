import numpy as np

# Linear Regression from Here.
# h_j(Q) := h_j(Q) - \alpha * d/dQ_j (J(Q))
class LinearRegression():
    '''
    theta = [0, 0]
    alpha = learning rate
    m = no of training examples
    xs = data for x
    ys = data for y
    '''
    theta, alpha, m, xs, ys = np.array([0, 0]), 0.01, 100, [], []
    gradientData = []
    threshold = 1e-12
    
    def __init__(self, theta, alpha, m, xs, ys,):
        self.theta = theta
        self.alpha = alpha
        self.m = m
        self.xs = xs
        self.ys = ys
        
    def step(self):
        s1, s2 = 0, 0
        for i in range(self.m):
            s1 += (self.theta[0] + self.theta[1] * self.xs[i] - self.ys[i])
            s2 += (self.theta[0] + self.theta[1] * self.xs[i] - self.ys[i]) * self.xs[i]
        return np.array([s1, s2])

    def J(self, Q):
        sum = 0
        for i in range(self.m):
            sum += (Q[0] + Q[1] * self.xs[i] - self.ys[i]) * (Q[0] + Q[1] * self.xs[i] - self.ys[i])
        return sum / (2 * self.m)
    
    def gradient_descent(self):
        threshold = self.threshold
        while True:
            ntheta = [self.theta[0], self.theta[1]]
            self.theta = self.theta - (self.alpha / self.m) * self.step() 
            self.gradientData.append((self.theta, self.J(self.theta)))
            if (abs(self.J(self.theta) - self.J(ntheta)) < threshold):
                break
            
        return self.theta

