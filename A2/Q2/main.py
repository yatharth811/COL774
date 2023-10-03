from cvxopt import matrix, solvers
import os
import cv2
import pandas as pd
import numpy as np
import time

# df = pd.DataFrame(columns=['features', 'label'])
features = []
y = []
def add_data(path: str, label: int):
  global df
  global y
  global features
  for file in os.listdir(path):
    image = cv2.imread(f'{path}/{file}')
    features.append(np.array(image).reshape(-1) / 255)
    # new_row = pd.DataFrame({'features': [features], 'label': [label]})
    # df = pd.concat([df, new_row], ignore_index=True)
    y.append(label)
    
add_data('Q2/resized_train_3', -1)
add_data('Q2/resized_train_4', 1)

# m = df.shape[0]
features = np.array(features)
m = len(features)
start = time.time()
# P = np.zeros((m, m))
# for i in range(m):
#   for j in range(m):
#     P[i][j] = np.dot(features[i], features[j]) * y[i] * y[j]
    
# q = matrix(-np.ones(m))
# G = matrix(np.vstack((-np.eye(m), np.eye(m))))
# h = matrix(np.hstack((np.zeros(m), np.ones(m))))
# A = matrix(y, (1, m), 'd')
# b = matrix(0.0)

P = matrix(np.outer(y, y) * np.dot(features, features.T))
q = matrix(-np.ones(m))
G = matrix(np.vstack((-np.eye(m), np.eye(m))))
h = matrix(np.hstack((np.zeros(m), np.ones(m))))
A = matrix(y, (1, m), 'd')
b = matrix(0.0)

# Solve the quadratic programming problem
solvers.options['show_progress'] = False
solution = solvers.qp(P, q, G, h, A, b)

# print("Time taken: ", time.time() - start)
# sol = solvers.qp(P, q, G, h, A, b)
# alphas = np.array(sol['x'])
# print(alphas)

alphas = np.ravel(solution['x'])

# Find support vectors (non-zero alphas)
support_vector_indices = np.where(alphas > 1e-4)[0]

print(f'Number of support Vectors are {len(support_vector_indices)}')
print(f'percentage of training samples that are support Vectors are {len(support_vector_indices)*100/m}')
# support_vectors = data[support_vector_indices]
# support_vector_labels = labels[support_vector_indices]

# Calculate the weight vector w
w = np.sum((alphas[i] * y[i] * features[i]) for i in support_vector_indices)

# Calculate the intercept term b
b = 0
for i in support_vector_indices:
    b += y[i]
    b -= np.dot(w, features[i])
b /= len(support_vector_indices)

print(w)
print(b)
