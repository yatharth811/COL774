import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
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

# Train
cross_validation_accuracies = []
validation_accuracies = []

C = [1e-5, 1e-3, 1, 5, 10]
for c in C:
  svc = SVC(C=c, kernel='rbf', gamma=0.001)
  score = np.mean(cross_val_score(svc, x_train, y_train, cv=5))
  cross_validation_accuracies.append(score * 100)
  print(f'C = {c}')
  print(f'5-Fold Cross Validation Accuracy: {score * 100}%')
  svc.fit(x_train, y_train)
  score = svc.score(x_test, y_test)
  validation_accuracies.append(score * 100)
  print(f'Validation Accuracy: {score * 100}%')
  
  
plt.figure(figsize=(10, 6))
plt.semilogx(C, cross_validation_accuracies, label='5-Fold CV Accuracy')
plt.semilogx(C, validation_accuracies, label='Validation Set Accuracy')
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.title('5-Fold Cross-Validation and Validation Set Accuracy vs. C')
plt.legend()
plt.grid(True)
plt.savefig('d.png')
plt.show()

print(cross_validation_accuracies, validation_accuracies)