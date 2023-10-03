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

sigma = np.array([np.zeros([2, 2]), np.zeros([2, 2])])
for i in range(0, m):
  x = np.array([[xs[i], ys[i]]])
  y = [u[zs[i]]]
  if (zs[i] == 0):
    sigma[0] += (x - y) * (x - y).T
  else:
    sigma[1] += (x - y) * (x - y).T

sigma[0], sigma[1] = sigma[0]/c0, sigma[1]/c1

print("u0: ", u[0])
print("u1: ", u[1])
print("phi: ", phi)
print("Sigma 0:\n", sigma[0])
print("Sigma 1:\n", sigma[1])