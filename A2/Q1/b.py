# https://web.stanford.edu/~jurafsky/slp3/4.pdf
import random
import pandas as pd
import time

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
  for i in range(df.shape[0]):
    cnt += 1 if random.choice(LABELS) == df.Sentiment[i] else 0
  return cnt / df.shape[0]
  
def positive_test(df):
  cnt = 0
  for i in range(df.shape[0]):
    cnt += 1 if POSITIVE == df.Sentiment[i] else 0
  return cnt / df.shape[0]
  
print("Random Heuristic:")
df = read_data('Q1/Corona_train.csv')
print("Train Accuracy: ", random_test(df))
df = read_data('Q1/Corona_validation.csv')
print("Test Accuracy: ", random_test(df))
print()
print("Positve Heuristic:")
df = read_data('Q1/Corona_train.csv')
print("Train Accuracy: ", positive_test(df))
df = read_data('Q1/Corona_validation.csv')
print("Test Accuracy: ", positive_test(df))

