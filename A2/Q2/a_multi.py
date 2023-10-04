from cvxopt import matrix, solvers
import os
import cv2
import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

features = []
def add_data(path: str):
  global features
  feature = []
  for file in os.listdir(path):
    image = cv2.imread(f'{path}/{file}')
    feature.append(np.array(image).reshape(-1) / 255)
  features.append(feature)
 
add_data('Q2/resized_train_0')
add_data('Q2/resized_train_1')
add_data('Q2/resized_train_2')
add_data('Q2/resized_train_3')
add_data('Q2/resized_train_4')
add_data('Q2/resized_train_5')

# m = df.shape[0]
# Reference: https://xavierbourretsicotte.github.io/SVM_implementation.html
# features = np.array(features)
# y = np.array(y)

def gaussian_kernel_matrix(X, gamma):
  distances_squared = -2 * np.dot(X, X.T) + np.sum(X ** 2, axis=1)[:, np.newaxis] + np.sum(X ** 2, axis=1)
  kernel_matrix = np.exp(-distances_squared * gamma)
  return kernel_matrix

def svm_classifier(label1: int, label2: int):
  y = [-1 for _ in range(len(features[label1]))] + [1 for _ in range(len(features[label1]))]
  y = np.array(y)
  
  feature = np.array(features[label1] + features[label2])
  m = len(feature)
  K = gaussian_kernel_matrix(X=feature, gamma=0.001)
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

  # print(f'Count of support vectors = ', len(support_vector_indices))
  # print(f'Support Vector % = ', (len(support_vector_indices) / m)  * 100 )

  calculate = lambda y, alphas, support_vector_indices, K: (
      sum(y[i] for i in support_vector_indices) -
      sum(alphas[j] * y[j] * K[i, j] for i in support_vector_indices for j in support_vector_indices)
  ) / len(support_vector_indices)
  b = calculate(y, alphas, support_vector_indices, K)
  # print(b)
  return [alphas, support_vector_indices, b, feature, y]

svms = {}
for i in range(6):
  for j in range(i + 1, 6):
    svms[(i, j)] = svm_classifier(i, j);
# svms = [[svm_classifier(i, j) for j in range(i + 1, 6)] for i in range(6)]

y_pred, y_test = [], []
index, cnt = 0, 0
def test(path: str, label: int):
  global cnt
  global index
  global y_pred
  global y_test
  for file in os.listdir(path):
    counter = defaultdict(int)
    image = cv2.imread(f'{path}/{file}')
    f = np.array(image).reshape(-1) / 255
    for i in range(6):
      for j in range(i + 1, 6):
        alphas, support_vector_indices, b, feature, y = svms[(i, j)]
        s = 0
        for si in support_vector_indices:
          s += alphas[si] * y[si] * np.exp(-0.001 * np.linalg.norm(f - feature[si]) ** 2)
        if (np.sign(s + b) == -1):
          counter[i] += 1
        else:
          counter[j] += 1
    
    predicted_label = max(counter, key=counter.get)
    cnt += 1 if (predicted_label == label) else 0
    index += 1
    y_pred.append(predicted_label)
    y_test.append(label)
  
test("Q2/resized_test_0", 0)
test("Q2/resized_test_1", 1)
test("Q2/resized_test_2", 2)
test("Q2/resized_test_3", 3)
test("Q2/resized_test_4", 4)
test("Q2/resized_test_5", 5)
print("Accuracy: ", (cnt / index) * 100)
print("Confusion Matrix:\n ", confusion_matrix(y_test, y_pred))
