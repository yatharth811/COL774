from utils import *
from LogisticRegression import LogisticRegression
xs = load_data("data/logisticX.csv", 0)
xs = normalise(xs)
ys = load_data("data/logisticX.csv", 1)
ys = normalise(ys)
zs = load_data("data/logisticY.csv")

model = LogisticRegression(xs = xs, ys = ys, zs = zs, theta = np.array([0, 0, 0]))
print(model.newton())