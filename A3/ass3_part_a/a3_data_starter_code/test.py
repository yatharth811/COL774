import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from pandas import read_csv, DataFrame, concat

label_encoder = None

def get_np_array(file_name):
    global label_encoder
    data = read_csv(file_name)

    need_label_encoding = ['team', 'host', 'opp', 'month', 'day_match']
    if label_encoder is None:
        label_encoder = OrdinalEncoder()
        label_encoder.fit(data[need_label_encoding])
    data_1 = DataFrame(label_encoder.transform(data[need_label_encoding]), columns=label_encoder.get_feature_names_out())

    # Merge the two dataframes
    dont_need_label_encoding = ["year", "toss", "bat_first", "format", "fow", "score", "rpo", "result"]
    data_2 = data[dont_need_label_encoding]
    final_data = concat([data_1, data_2], axis=1)

    X = final_data.iloc[:, :-1]
    y = final_data.iloc[:, -1:]
    return X.to_numpy(), y.to_numpy()

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.children = []  # Initialize an empty list for children

    def entropy(self, y):
        # Calculate the entropy of a set of labels y
        unique_classes, class_counts = np.unique(y, return_counts=True)
        probabilities = class_counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def information_gain(self, y, splits):
        # Calculate information gain after a split
        H_S = self.entropy(y)
        total_size = sum(len(split) for split in splits)
        C_S = 0

        for split in splits:
            C_S += (len(split) / total_size) * self.entropy(split)

        return H_S - C_S

    def find_best_split(self, X, y, types):
        best_feature = None
        best_splits = None
        best_score = 0

        H_S = self.entropy(y)

        for feature in range(X.shape[1]):
            if types[feature] == 'cat':
                unique_values = np.unique(X[:, feature])
                splits = []
                for value in unique_values:
                    mask = X[:, feature] == value
                    splits.append(y[mask])

                info_gain = self.information_gain(y, splits)
                if info_gain > best_score:
                    best_feature = feature
                    best_splits = splits
                    best_score = info_gain

            else:  # Continuous variable
                values = np.median(X[:, feature])
                left_mask = X[:, feature] <= values
                right_mask = X[:, feature] > values
                left_y = y[left_mask]
                right_y = y[right_mask]
                splits = [left_y, right_y]
                info_gain = self.information_gain(y, splits)
                if info_gain > best_score:
                    best_feature = feature
                    best_value = values
                    best_splits = splits
                    best_score = info_gain

        return best_feature, best_splits

    def fit(self, X, y, types, depth=0):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            self.tree = np.argmax(np.bincount(y))
        else:
            best_feature, best_splits = self.find_best_split(X, y, types)
            if best_feature is not None:
                self.tree = (best_feature, None)
                self.children = []
                for i, split in enumerate(best_splits):
                    child = DecisionTree(max_depth=self.max_depth)
                    child.fit(X[split], y[split], types, depth + 1)
                    self.children.append(child)
            else:
                self.tree = np.argmax(np.bincount(y))

    def predict(self, X):
        if isinstance(self.tree, int):
            return self.tree
        feature, value = self.tree
        if types[feature] == 'cat':
            for i, unique_value in enumerate(np.unique(X[:, feature])):
                if X[feature] == unique_value:
                    return self.children[i].predict(X)
        else:
            if X[feature] <= value:
                return self.children[0].predict(X)
            else:
                return self.children[1].predict(X)
        return None

if __name__ == '__main__':
    X_train, y_train = get_np_array('train.csv')
    y_train = np.array([y[0] for y in y_train])
    X_test, y_test = get_np_array("test.csv")
    y_test = np.array([y[0] for y in y_test])
    types = ['cat', 'cat', 'cat', "cat", "cat", 'cont', 'cat', 'cat', 'cat', 'cont', 'cont', 'cont']
    max_depth = 5
    tree = DecisionTree(max_depth)
    tree.fit(X_train, y_train, types)
    train_correct = 0
    for i in range(X_train.shape[0]):
        train_correct += (tree.predict(X_train[i]) == y_train[i])
    print(f"Train Accuracy: {train_correct / X_train.shape[0] * 100}%")

    test_correct = 0
    for i in range(X_test.shape[0]):
        test_correct += (tree.predict(X_test[i]) == y_test[i])
    print(f"Test Accuracy: {test_correct / X_test.shape[0] * 100}%")
