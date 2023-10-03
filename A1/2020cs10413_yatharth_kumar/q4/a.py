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

print("u0: ", u[0])
print("u1: ", u[1])
print("phi: ", phi)
print("Sigma:\n", sigma)