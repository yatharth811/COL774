from utils import *
from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt

xs = load_data("data/logisticX.csv", 0)
xs = normalise(xs)
ys = load_data("data/logisticX.csv", 1)
ys = normalise(ys)
zs = load_data("data/logisticY.csv")

model = LogisticRegression(xs = xs, ys = ys, zs = zs, theta = np.array([0, 0, 0]))
theta = model.newton()

markers = ['o', 'x'] 
colors = ['red', 'blue']
m = len(xs)
for i in range(m):
  x, y, z = xs[i], ys[i], int(zs[i])
  plt.scatter(x, y, marker=markers[z], c=colors[z])

plt.scatter(xs[0], ys[0], marker='o', label='Class 0', c='red')
plt.scatter(xs[-1], ys[-1], marker='x', label='Class 1', c='blue')
p_x = np.linspace(-3, 3, 100)
p_y = -1 * (theta[0] + theta[1] * p_x) / theta[2]
plt.plot(p_y, p_x, color='black', label='Logistic Regression Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Logistic Regression')
plt.legend()
plt.show()