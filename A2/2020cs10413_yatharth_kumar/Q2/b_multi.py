import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import cv2
import os
import matplotlib.pyplot as plt

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
    
add_data('Q2/resized_train_0', 0)
add_data('Q2/resized_train_1', 1)
add_data('Q2/resized_train_2', 2)
add_data('Q2/resized_train_3', 3)
add_data('Q2/resized_train_4', 4)
add_data('Q2/resized_train_5', 5)

# Train
svc = SVC(C=1.0, kernel='rbf', gamma=0.001)
svc.fit(x_train, y_train)

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
    
read_test("Q2/resized_test_0", 0)
read_test("Q2/resized_test_1", 1)
read_test("Q2/resized_test_2", 2)
read_test("Q2/resized_test_3", 3)
read_test("Q2/resized_test_4", 4)
read_test("Q2/resized_test_5", 5)

y_pred = svc.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {score * 100}%")

cmatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cmatrix)
report = classification_report(y_test, y_pred)
print("Report:")
print(report)

mislabeled = []
for i, y in enumerate(y_pred):
  if (y != y_test[i]):
    mislabeled.append((i, y))
  
  if (len(mislabeled) == 12):
    break
  
  
for i in range(12):
  index, label = mislabeled[i]
  contents = os.listdir(f'Q2/val/{y_test[index]}')
  image = cv2.imread(f'Q2/val/{y_test[index]}/{contents[index % len(contents)]}')
  plt.imshow(image)
  plt.title(f'Predicted Label: {label}, Correct Label: {y_test[index]}')
  plt.savefig(f'b_{i}.png')
  plt.show()