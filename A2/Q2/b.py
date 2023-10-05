from cvxopt import matrix, solvers
import os
import cv2
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

features = []
y = []
def add_data(path: str, label: int):
  global df
  global y
  global features
  for file in os.listdir(path):
    image = cv2.imread(f'{path}/{file}')
    features.append(np.array(image).reshape(-1) / 255)
    y.append(label)
    
add_data('Q2/resized_train_3', -1)
add_data('Q2/resized_train_4', 1)

# m = df.shape[0]
# Reference: https://xavierbourretsicotte.github.io/SVM_implementation.html
features = np.array(features)
y = np.array(y)

def gaussian_kernel_matrix(X, gamma):
  distances_squared = -2 * np.dot(X, X.T) + np.sum(X ** 2, axis=1)[:, np.newaxis] + np.sum(X ** 2, axis=1)
  kernel_matrix = np.exp(-distances_squared * gamma)
  return kernel_matrix

m = len(features)
K = gaussian_kernel_matrix(X=features, gamma=0.001)
P = matrix(np.outer(y, y) * K)
q = matrix(-np.ones(m))
G = matrix(np.vstack((-np.eye(m), np.eye(m))))
h = matrix(np.hstack((np.zeros(m), np.ones(m))))
A = matrix(y, (1, m), 'd')
b = matrix(0.0)

solvers.options['show_progress'] = False
solution = solvers.qp(P, q, G, h, A, b)
alphas = np.ravel(solution['x'])

# Find support vectors (non-zero alphas)
support_vector_indices = np.where(alphas > 1e-4)[0]

print(f'Count of support vectors = ', len(support_vector_indices))
print(f'Support Vector % = ', (len(support_vector_indices) / m)  * 100 )

calculate = lambda y, alphas, support_vector_indices, K: (
    sum(y[i] for i in support_vector_indices) -
    sum(alphas[j] * y[j] * K[i, j] for i in support_vector_indices for j in support_vector_indices)
) / len(support_vector_indices)
b = calculate(y, alphas, support_vector_indices, K)
print(b)


index, cnt = 0, 0
def test(path: str, label: int):
  global cnt
  global index
  for file in os.listdir(path):
    image = cv2.imread(f'{path}/{file}')
    f = np.array(image).reshape(-1) / 255
    s = 0
    for si in support_vector_indices:
      s += alphas[si] * y[si] * np.exp(-0.001 * np.linalg.norm(f - features[si]) ** 2)
      
    cnt += 1 if (np.sign(s + b) == label) else 0
    index += 1
  
test("Q2/resized_test_3", -1)
test("Q2/resized_test_4", 1)
print("Accuracy: ", (cnt / index) * 100)


# Top 6
nalphas = sorted(alphas, reverse=True)
alps = [nalphas[i] for i in range(6)]
image_data = [np.reshape(features[i], (16, 16, 3)) for i in range(6)]

fig, axes = plt.subplots(2, 3)
for i, ax in enumerate(axes.ravel()):
  ax.imshow(image_data[i])
  ax.set_title(f"Alpha: {alps[i]:.2f}")

plt.tight_layout()
plt.savefig('top6_b.png')
plt.show()
