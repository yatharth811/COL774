# https://web.stanford.edu/~jurafsky/slp3/4.pdf
import random
import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix

NEUTRAL = 'Neutral'
POSITIVE = 'Positive'
NEGATIVE = 'Negative'
LABELS = [NEUTRAL, POSITIVE, NEGATIVE]
random.seed(time.time())

def read_data(path: str):
  reader = pd.read_csv(path)
  reader['CoronaTweet'] = reader['CoronaTweet'].str.lower()
  return reader
  
def random_test(df):
  cnt = 0
  y_pred = []
  for i in range(df.shape[0]):
    predicted_label = random.choice(LABELS)
    y_pred.append(predicted_label)
    cnt += 1 if predicted_label == df.Sentiment[i] else 0
  return (cnt / df.shape[0]), y_pred
  
def positive_test(df):
  cnt = 0
  y_pred = []
  for i in range(df.shape[0]):
    cnt += 1 if POSITIVE == df.Sentiment[i] else 0
    y_pred.append(POSITIVE)
  return (cnt / df.shape[0]), y_pred
  
print("Random Heuristic:")
df = read_data('Q1/Corona_validation.csv')
accuracy, y_pred = random_test(df)
print("Test Accuracy: ", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(np.array(df['Sentiment']), y_pred))

df = read_data('Q1/Corona_train.csv')
accuracy, y_pred = random_test(df)
print("Train Accuracy: ", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(np.array(df['Sentiment']), y_pred))

print()

print("Positve Heuristic:")
df = read_data('Q1/Corona_validation.csv')
accuracy, y_pred = positive_test(df)
print("Test Accuracy: ", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(np.array(df['Sentiment']), y_pred))

df = read_data('Q1/Corona_train.csv')
accuracy, y_pred = positive_test(df)
print("Train Accuracy: ", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(np.array(df['Sentiment']), y_pred))

