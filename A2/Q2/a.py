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
m = len(features)
P = matrix(np.outer(y, y) * np.dot(features, features.T))
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

w, b = 0, 0
for index in support_vector_indices:
  w += alphas[index] * y[index] * features[index]

for index in support_vector_indices:
    b += y[index] - np.dot(w, features[index])
b /= len(support_vector_indices)

print("Weight with linear kernel: ", w)
print("Bias with linear kernel: ", b)

index, cnt = 0, 0
def test(path: str, label: int):
  global cnt
  global index
  for file in os.listdir(path):
    image = cv2.imread(f'{path}/{file}')
    # features.append()
    cnt += 1 if (np.sign(np.dot(np.array(image).reshape(-1) / 255, w) + b) == label) else 0
    index += 1
    # print(np.sign(np.dot(np.array(image).reshape(-1) / 255, w) + b), np.dot(np.array(image).reshape(-1) / 255, w) + b)
  
test("Q2/resized_test_3", -1)
test("Q2/resized_test_4", 1)
# print(index, m)
print("Accuracy: ", (cnt / index) * 100)
# print(cnt, index)


# plot top 6 coefficients and weight vector 
nalphas = sorted(alphas, reverse=True)
alps = [nalphas[i] for i in range(6)]
image_data = [np.reshape(features[i], (16, 16, 3)) for i in range(6)]
image_data.append(np.reshape(w, (16, 16, 3)))

fig, axes = plt.subplots(2, 4)
for i, ax in enumerate(axes.ravel()):
    if i < 7:
        ax.imshow(image_data[i])
        ax.set_title(f"Alpha: {alps[i]:.2f}") if i < 6 else ax.set_title(f"Weight, w")
    else:
        ax.axis("off")

plt.tight_layout()
plt.savefig('top6_a.png')
plt.show()