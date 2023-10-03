import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
import os

x_train = []
y_train = []
def add_data(path: str, label: int):
  global df
  global y
  global x_train
  for file in os.listdir(path):
    image = cv2.imread(f'{path}/{file}')
    x_train.append(np.array(image).reshape(-1) / 255)
    y_train.append(label)
    
add_data('Q2/resized_train_3', -1)
add_data('Q2/resized_train_4', 1)

# Train
lsvc = SVC(C=1.0, kernel='linear')
# lsvc = LinearSVC(verbose=0)
lsvc.fit(x_train, y_train)


# Test
x_test= []
y_test = []
def read_test(path: str, label: int):
  global x_test
  global y_test
  for file in os.listdir(path):
    image = cv2.imread(f'{path}/{file}')
    x_test.append(np.array(image).reshape(-1) / 255)
    y_test.append(label)
    
read_test("Q2/resized_test_3", -1)
read_test("Q2/resized_test_4", 1)

score = lsvc.score(x_test, y_test)
print('Number of Support Vectors with linear kernel', sum(lsvc.n_support_))
print('Weight with linear kernel = ', lsvc.coef_)
print('Bias with linear kernel = ', lsvc.intercept_)
print(f"Accuracy: {score * 100}%")