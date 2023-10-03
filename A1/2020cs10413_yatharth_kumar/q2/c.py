import numpy as np
import csv

thetas = [np.array([3.00628859, 0.95340697, 2.03434063]),
         np.array([2.97696983, 1.00400722, 1.99806896]),
         np.array([2.95488385, 1.07485514, 1.97378174]),
         np.array([2.9940582, 1.0408797, 2.01411928])]

data = []
rows = 0
with open('data/q2test.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        data.append([1.0] + row)
        rows += 1
        
data= np.array(data, dtype=np.float64)

def J(theta):
    error=0
    for i in range(rows):
        val = theta[0] * data[i][0] + theta[1]*data[i][1] + theta[2]*data[i][2] - data[i][3]
        error += val * val
    return error / (2 * rows)

print(J(np.array([3.0,1.0,2.0])))
for i in range(0, 4):
  print(J(thetas[i]))