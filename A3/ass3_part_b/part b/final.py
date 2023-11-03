import numpy as np 
import sys
import pdb
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
import copy
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def get_data(x_path, y_path):
    '''
    Args:
        x_path: path to x file
        y_path: path to y file
    Returns:
        x: np array of [NUM_OF_SAMPLES x n]
        y: np array of [NUM_OF_SAMPLES]
    '''
    x = np.load(x_path)
    y = np.load(y_path)

    y = y.astype('float')
    x = x.astype('float')

    #normalize x:
    x = 2*(0.5 - x/255)
    return x, y

def get_metric(y_true, y_pred):
    '''
    Args:
        y_true: np array of [NUM_SAMPLES x r] (one hot) 
                or np array of [NUM_SAMPLES]
        y_pred: np array of [NUM_SAMPLES x r] (one hot) 
                or np array of [NUM_SAMPLES]
                
    '''
    results = classification_report(y_true, y_pred)
    print(results)
    avg_score = f1_score(y_true,y_pred,average='macro')
    print(avg_score)
    return avg_score


def softmax(data):
    max_vals = np.max(data, axis=1, keepdims=True)
    softmax_data = np.exp(data - max_vals) / np.sum(np.exp(data - max_vals), axis=1, keepdims=True)
    return softmax_data

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1 - z)


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return 1. * (z > 0)


class MLPClassifier:
    def __init__(self, X, Y, layers, activation, activation_derivative,
                 learning_rate=0.1, epsilon=1e-8, batch_size=100, max_epochs=1000, adaptive=False, verbose=False):
        permutation = np.random.permutation(X.shape[0])
        self.X = X[permutation]
        self.Y = Y[permutation]
        self.layers = [X.shape[1]] + layers + [Y.shape[1]]
        self.weights = []
        self.biases = []
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.learning_rate = learning_rate
        if adaptive:
            self.learning_rate *= 10
        self.epsilon = epsilon
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.adaptive = adaptive
        self.verbose = verbose
        self.init_weights_biases()
        print(self.weights[0].shape)

    def init_weights_biases(self):
        for i in range(1, len(self.layers)):
            self.weights.append(np.random.randn(
                self.layers[i], self.layers[i - 1]) * (2 / self.layers[i - 1]) ** 0.5)
            self.biases.append(np.random.randn(
                self.layers[i], 1) * (2 / self.layers[i - 1]) ** 0.5)

    def forward_propagation(self, X):
        self.a = [X]
        for i in range(len(self.layers) - 1):
            z = np.matmul(self.a[i], self.weights[i].T) + self.biases[i].T
            if i == len(self.layers) - 2:
                self.a.append(softmax(z))
            else:
                self.a.append(self.activation(z))
        return self.a[-1]

    def back_propagation(self, Y):
        self.delta = [(Y - self.a[-1]) *
                      sigmoid_derivative(self.a[-1])]
        for i in range(len(self.layers) - 2, 0, -1):
            self.delta.append(
                np.matmul(self.delta[-1], self.weights[i]) * self.activation_derivative(self.a[i]))
        self.delta.reverse()

    def update_weights_biases(self, i):
        rate = self.learning_rate
        if self.adaptive:
            rate = self.learning_rate / i ** 0.5
        for i in range(len(self.layers) - 1):
            self.weights[i] += rate * \
                np.matmul(self.delta[i].T, self.a[i]) / self.batch_size
            self.biases[i] += rate * \
                np.sum(self.delta[i], axis=0,
                       keepdims=True).T / self.batch_size

    def train(self):
        prev_cost, curr_cost, i = 1e9, 0, 0
        while abs(curr_cost - prev_cost) > self.epsilon and i < self.max_epochs:
            prev_cost = curr_cost
            for j in range(0, len(self.X), self.batch_size):
                X = self.X[j:j + self.batch_size]
                Y = self.Y[j:j + self.batch_size]
                nn_out = self.forward_propagation(X)
                self.back_propagation(Y)
                self.update_weights_biases(i + 1)
                curr_cost += self.cost(Y, nn_out)
            curr_cost /= len(self.X) / self.batch_size
            i += 1
            if self.verbose:
                print(f'Epoch: {i}, Cost: {curr_cost}')

    def predict(self, X):
        probabilities = self.forward_propagation(X)
        max_indices = np.argmax(probabilities,axis=1)
        result = np.zeros_like(probabilities)
        result[np.arange(X.shape[0]), max_indices] = 1
        return result

    def cost(self, X, Y):
        return log_loss(X,Y)


if __name__ == '__main__':

    x_train_path = "x_train.npy"
    y_train_path = "y_train.npy"

    X_train, y_train = get_data(x_train_path, y_train_path)

    x_test_path = "x_test.npy"
    y_test_path = "y_test.npy"

    X_test, y_test = get_data(x_test_path, y_test_path)

    #you might need one hot encoded y in part a,b,c,d,e
    label_encoder = OneHotEncoder(sparse_output = False)
    label_encoder.fit(np.expand_dims(y_train, axis = -1))

    y_train_onehot = label_encoder.transform(np.expand_dims(y_train, axis = -1))
    y_test_onehot = label_encoder.transform(np.expand_dims(y_test, axis = -1))

    # print(X_train.shape)
    # print(y_train_onehot.shape)

    clf = MLPClassifier(X_train, y_train_onehot,[512,256], sigmoid, sigmoid_derivative, learning_rate=0.01, max_epochs=500, adaptive=True)



    nns = [[512],[512,256],[512,256,128],[512,256,128,64]]
    f1_scores = []
    accuracies = []
    network_depth = []
    for nn in nns:
        clf = MLPClassifier(X_train, y_train_onehot,nn, sigmoid, sigmoid_derivative, 
                            learning_rate=0.01, max_epochs=500, adaptive=True)
        clf.train()
        y_pred = clf.predict(X_test)
        f1_scores.append(get_metric(y_test_onehot,y_pred))
        network_depth.append(len(nn))
        # print(y_test_onehot.shape)
        # print(y_pred.shape)
        accuracies.append(accuracy_score(y_test_onehot,y_pred))

    plt.figure(1)
    plt.plot(network_depth,f1_scores)
    plt.xlabel("Network Depth")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Network Depth")
    plt.savefig('d_1')    

    plt.figure(2)
    plt.plot(network_depth,accuracies)
    plt.xlabel("Network Depth")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Network Depth")
    plt.savefig('d_2') 