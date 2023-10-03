from utils import *
import matplotlib.pyplot as plt
xs = load_data_X("data/q4x.dat", 0)
xs = normalise(xs)
ys = load_data_X("data/q4x.dat", 1)
ys = normalise(ys)
zs = load_data_Y("data/q4y.dat")

markers = ['o', 'x'] 
colors = ['red', 'blue']
for i, _ in enumerate(xs):
  plt.scatter(xs[i], ys[i], marker=markers[zs[i]], c=colors[zs[i]])

plt.scatter(xs[0], ys[0], marker=markers[0], c=colors[0], label="Alaska")
plt.scatter(xs[-1], ys[-1], marker=markers[1], c=colors[1], label="Canada")
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()