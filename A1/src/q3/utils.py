import csv
import numpy as np
from matplotlib import pyplot as plt

def load_data(dir, col = 0):
	f = open(dir, "r")
	data = list(csv.reader(f))
	f.close()
	return np.array([float(x[col]) for x in data])

def normalise(lst):
  si = np.std(lst)
  mu = np.mean(lst)
  return np.array([(x - mu)/si for x in lst])