import numpy as np
from matplotlib import pyplot as plt

def load_data_X(dir, col):
  f = open(dir, "r")
  data = [float(x.split()[col]) for x in f.readlines()]
  f.close()
  return np.array(data)

def load_data_Y(dir):
  f = open(dir, "r")
  data = [0 if x == "Alaska\n" else 1 for x in f.readlines()]
  f.close()
  return np.array(data)

def normalise(lst):
  si = np.std(lst)
  mu = np.mean(lst)
  return np.array([(x - mu)/si for x in lst])