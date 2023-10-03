from utils import *

xs = load_data_X("data/q4x.dat", 0)
xs = normalise(xs)
ys = load_data_X("data/q4x.dat", 1)
ys = normalise(ys)
zs = load_data_Y("data/q4y.dat")

m = len(xs)
u = np.zeros([2, 2])
c0, c1 = 0, 0
for i in range(0, m):
  if (zs[i] == 0):
    u[0] += np.array([xs[i], ys[i]])
    c0 += 1
  else:
    u[1] += np.array([xs[i], ys[i]])
    c1 += 1

phi = c1/m
u[0] /= c0
u[1] /= c1

sigma = np.zeros([2, 2])
for i in range(0, m):
  x = np.array([[xs[i], ys[i]]])
  y = [u[zs[i]]]
  sigma += (x - y) * (x - y).T

sigma = sigma / m
sigma_inverse = np.linalg.inv(sigma)

linear_boundary = lambda x1: (
    -((sigma_inverse[0][0] * u[0][0] + sigma_inverse[0][1] * u[0][1]) * x1) /
    (sigma_inverse[1][0] * u[0][0] + sigma_inverse[1][1] * u[0][1])
)

data_x = np.linspace(min(xs), max(xs), 100)
data_y = [linear_boundary(x) for x in data_x]

markers = ['o', 'x'] 
colors = ['red', 'blue']
for i, _ in enumerate(xs):
  plt.scatter(xs[i], ys[i], marker=markers[zs[i]], c=colors[zs[i]])

plt.scatter(xs[0], ys[0], marker=markers[0], c=colors[0], label="Alaska")
plt.scatter(xs[-1], ys[-1], marker=markers[1], c=colors[1], label="Canada")
plt.plot(data_x, data_y, c='Black', label='Linear Boundary')
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()