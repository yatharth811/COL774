from sklearn.preprocessing import OrdinalEncoder
from pandas import read_csv, DataFrame, concat
import numpy as np
import matplotlib.pyplot as plt
label_encoder = None 

def get_np_array(file_name):
  global label_encoder
  data = read_csv(file_name)
  
  need_label_encoding = ['team','host','opp','month', 'day_match']
  if label_encoder is None:
    label_encoder = OrdinalEncoder()
    label_encoder.fit(data[need_label_encoding])
  data_1 = DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
  
  #merge the two dataframes
  dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
  data_2 = data[dont_need_label_encoding]
  final_data = concat([data_1, data_2], axis=1)
  
  X = final_data.iloc[:,:-1]
  y = final_data.iloc[:,-1:]
  return X.to_numpy(), y.to_numpy()


class DecisionTree:
  
  # pass this to each node
  def __init__(self, max_depth: int):
    self.max_depth = max_depth
    self.tree = None
  
  # Recursive code to fit the training examples
  def fit(self, X, y, types):
    if (self.max_depth == 0):
      self.tree = int(np.argmax(np.bincount(y)))
    else:
      best_feature, best_value, is_continuous = self.find_best_split(X, y, types)
      
      if best_feature is None:
        self.tree = int(np.argmax(np.bincount(y)))
      else:
        Xs, ys = self.split_data_cont(X, y, best_feature, best_value) if is_continuous else self.split_data_cat(X, y, best_feature, best_value)
        self.tree = {
            "feature": best_feature,
            "value": best_value,
            "is_continuous": is_continuous,
            "children": [
              DecisionTree(self.max_depth - 1) for _ in range(len(Xs))
            ],
            "leaf_value": int(np.argmax(np.bincount(y)))
        }
        for i in range(len(Xs)):
          self.tree["children"][i].fit(Xs[i], ys[i], types)
  
  def find_best_split(self, X, y, types):
    best_feature = None
    best_value = None
    best_score = 0
    is_continuous = None
    H_S = self.entropy(y)
    
    for feature in range(X.shape[1]):
      if (types[feature] == 'cat'):
        # split the data in to categories based on the features aaaaa
        value = np.unique(X[:, feature])
        _, ys = self.split_data_cat(X, y, feature, value)
        ig = self.information_gain(ys, H_S)
        if (ig > best_score):
          best_feature = feature
          best_value = np.unique(X[:, feature])
          best_score = ig
          is_continuous = False
        
      else:
        # find median here
        value = np.median(X[:, feature])
        _, ys = self.split_data_cont(X, y, feature, value)
        ig = self.information_gain(ys, H_S)
        if (ig > best_score):
          best_feature = feature
          best_score = ig
          best_value = value
          is_continuous = True
      
    return best_feature, best_value, is_continuous
  
  def information_gain(self, ys, H_S):
    lens = [len(y) for y in ys]
    denominator = np.sum(lens)
    lens /= denominator
    C_S = 0.0
    for i, y in enumerate(ys):
      C_S += (lens[i]) * self.entropy(y)
    return H_S - C_S

  def entropy(self, y):
    if (len(y) == 0): 
      return 0
    p = np.mean(y)
    if p == 0 or p == 1:
      return 0
    return -p * np.log10(p) - (1.0 - p) * np.log10(1.0 - p)
  
  def split_data_cat(self, X, y, feature, categories):
    Xs, ys = [], []
    for category in categories:
      mask = X[:, feature] == category
      Xs.append(X[mask])
      ys.append(y[mask])
    
    # print("X: ", Xs)
    # print()
    # print()
    # return np.array(Xs), np.array(ys)
    return Xs, ys
  
  def split_data_cont(self, X, y, feature, value):
    mask = X[:, feature] <= value
    # Xs = np.array([X[mask], X[~mask]])
    # ys = np.array([y[mask], y[~mask]])
    Xs = [X[mask], X[~mask]]
    ys = [y[mask], y[~mask]]
    return Xs, ys
  
  def predict(self, X):
    if isinstance(self.tree, int):
      return self.tree
    else:
      feature = self.tree["feature"]
      value = self.tree["value"]
      if self.tree["is_continuous"]:
        if (X[feature] <= value):
          return self.tree["children"][0].predict(X)
        else:
          return self.tree["children"][1].predict(X)
      else:
        if X[feature] in value:
          for i, v in enumerate(value):
            if (v == X[feature]):
              return self.tree["children"][i].predict(X) 
        else:
          return self.tree["leaf_value"]


if __name__ == '__main__':
  X_train,y_train = get_np_array('train.csv')
  y_train = np.array([y[0] for y in y_train])
  X_test, y_test = get_np_array("test.csv")
  y_test = np.array([y[0] for y in y_test])
  types = ["cat","cat","cat","cat","cat","cont","cat","cat","cat","cont","cont","cont"]
  depths = [5, 10, 15, 20, 25]
  train_accuracies = []
  test_accuracies = []
  only_win_accuracies = []
  only_loss_accuracies = []
  for depth in depths:
    max_depth = depth
    tree = DecisionTree(max_depth = max_depth)
    tree.fit(X_train,y_train,types)
    train_correct = 0
    for i in range(X_train.shape[0]):
      train_correct += (tree.predict(X_train[i]) == y_train[i])
    train_accuracy = train_correct / X_train.shape[0] * 100
    
    test_correct = 0
    only_win_correct = 0
    only_loss_correct = 0
    for i in range(X_test.shape[0]):
      prediction = tree.predict(X_test[i]) 
      test_correct += (prediction == y_test[i])
      only_win_correct += (1 == y_test[i])   
      only_loss_correct += (0 == y_test[i])
      
    test_accuracy = test_correct / X_test.shape[0] * 100
    only_win_accuracy = only_win_correct / X_test.shape[0] * 100
    only_loss_accuracy = only_loss_correct / X_test.shape[0] * 100
    only_win_accuracies.append(only_win_accuracy)
    only_loss_accuracies.append(only_loss_accuracy)
    
    
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    print(f"Max Depth: {max_depth}")
    print(f"Train Accuracy: {train_accuracy}%")
    print(f"Test Accuracy: {test_accuracy}%")
    print(f"Only Win Accuracy: {only_win_accuracy}%")
    print(f"Only Loss Accuracy: {only_loss_accuracy}%")

    
  plt.plot(depths, train_accuracies, label='Training Accuracy')
  plt.plot(depths, test_accuracies, label='Test Accuracy')
  plt.plot(depths, only_win_accuracies, label='Only Win Accuracy')
  plt.plot(depths, only_loss_accuracies, label='Only Loss Accuracy')

  plt.xlabel("Maximum Depth of the Tree")
  plt.ylabel("Accuracy (in %)")
  plt.title('Decision Trees')
  plt.legend()
  plt.savefig('a.png')
  plt.show()