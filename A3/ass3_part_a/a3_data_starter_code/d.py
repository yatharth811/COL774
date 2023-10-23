from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from pandas import read_csv, DataFrame, concat
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np

one_hot_encoder = None

def get_np_array(file_name):
    global one_hot_encoder
    data = read_csv(file_name)
    
    need_one_hot_encoding = ['team', 'host', 'opp', 'month', 'day_match']
    
    if one_hot_encoder is None:
        one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        one_hot_encoder.fit(data[need_one_hot_encoding])
    
    one_hot_encoded_data = one_hot_encoder.transform(data[need_one_hot_encoding])
    
    one_hot_feature_names = one_hot_encoder.get_feature_names_out(input_features=need_one_hot_encoding)
    data_1 = DataFrame(one_hot_encoded_data, columns=one_hot_feature_names)
    
    # Merge the two dataframes
    dont_need_one_hot_encoding = ["year", "toss", "bat_first", "format", "fow", "score", "rpo", "result"]
    data_2 = data[dont_need_one_hot_encoding]
    final_data = concat([data_1, data_2], axis=1)
    
    X = final_data.iloc[:, :-1]
    y = final_data.iloc[:, -1:]
    
    return X.to_numpy(), y.to_numpy()


def sub_part_a():
  depths = [15, 25, 35, 45]
  train_accuracies = []
  test_accuracies = []
  X_train,y_train = get_np_array('train.csv')
  y_train = np.array([y[0] for y in y_train])

  X_test, y_test = get_np_array("test.csv")
  y_test = np.array([y[0] for y in y_test])


  for d in depths:
    model = DecisionTreeClassifier(criterion='entropy', max_depth=d)
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    
  plt.figure(figsize=(10, 5))
  plt.plot(depths, test_accuracies, label='Test Accuracies', marker='o')
  plt.plot(depths, train_accuracies, label='Train Accuracies', marker='o')
  plt.xlabel("Maximum Depth of Decision Tree")
  plt.ylabel("Accuracy")
  plt.title("Accuracy vs. Maximum Depth of Decision Tree")

  # Annotate the data points with accuracy values
  for x, test_acc, train_acc in zip(depths, test_accuracies, train_accuracies):
      plt.text(x, test_acc, f'Test Acc: {test_acc:.2f}', ha='center', va='bottom', color='blue')
      plt.text(x, train_acc, f'Train Acc: {train_acc:.2f}', ha='center', va='top', color='orange')

  # Create the legend
  plt.legend()
  plt.grid(True)
  plt.show()
  
  X_val, y_val = get_np_array("val.csv")
  y_val = np.array([y[0] for y in y_val])
  best_depth= 0
  best_accuracy = 0
  for d in depths:
    model = DecisionTreeClassifier(criterion='entropy', max_depth=d)
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"Depth: {d}, accuracy: {val_accuracy}")
    
    if (val_accuracy > best_accuracy):
      best_accuracy = val_accuracy
      best_depth = d
      
  return best_depth
  
def sub_part_b():
  alphas = [0.001, 0.01, 0.1, 0.2]
  train_accuracies = []
  test_accuracies = []
  X_train,y_train = get_np_array('train.csv')
  y_train = np.array([y[0] for y in y_train])

  X_test, y_test = get_np_array("test.csv")
  y_test = np.array([y[0] for y in y_test])

  for alpha in alphas:
    model = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha)
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    
  plt.figure(figsize=(10, 5))
  plt.plot(alphas, test_accuracies, label='Test Accuracies', marker='o')
  plt.plot(alphas, train_accuracies, label='Train Accuracies', marker='o')
  plt.xlabel("Pruning Parameters of Decision Tree")
  plt.ylabel("Accuracy")
  plt.title("Accuracy vs. Pruning Parameters of Decision Tree")

  # Annotate the data points with accuracy values
  for x, test_acc, train_acc in zip(alphas, test_accuracies, train_accuracies):
      plt.text(x, test_acc, f'Test Acc: {test_acc:.2f}', ha='center', va='top', color='blue')
      plt.text(x, train_acc, f'Train Acc: {train_acc:.2f}', ha='center', va='bottom', color='orange')

  # Create the legend
  plt.legend()
  plt.grid(True)
  plt.show()
  
  X_val, y_val = get_np_array("val.csv")
  y_val = np.array([y[0] for y in y_val])
  best_alpha = 0
  best_accuracy = 0
  for alpha in alphas:
    model = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha)
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    if (val_accuracy > best_accuracy):
      best_accuracy = val_accuracy
      best_alpha = alpha
      
  return best_alpha
  
  
best_depth = sub_part_a()
best_alpha = sub_part_b()

print(f"Best Depth: {best_depth}, Best Alpha: {best_alpha}")
X_train,y_train = get_np_array('train.csv')
y_train = np.array([y[0] for y in y_train])
X_test, y_test = get_np_array("test.csv")
y_test = np.array([y[0] for y in y_test])
best_model = DecisionTreeClassifier(criterion='entropy', ccp_alpha=best_alpha, max_depth=best_depth)
best_model.fit(X_train, y_train)
accuracy = best_model.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100}%")