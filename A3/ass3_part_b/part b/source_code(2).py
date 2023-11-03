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

sigmoid = lambda z: 1 / (1 + np.exp(-z))
sigmoid_derivative = lambda z: z * (1 - z)
relu = lambda z: np.maximum(0, z)
relu_derivative = lambda z: 1. * (z > 0)

class NeuralNetwork:
    def __init__(self, input_feature, output_feature, layers, activation='sigmoid',
                 learning_rate=0.1, eps=1e-8, batch_size=32, max_epochs=500, adaptive=False):
        self.layers = [input_feature, *layers, output_feature]
        self.wgt = []
        self.bs = []
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            self.activation = relu
            self.activation_derivative = relu_derivative
        self.lr = learning_rate
        self.eps = eps
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.adpt = adaptive
        self.wgt = [np.random.randn(self.layers[i], self.layers[i - 1]) * (2 / self.layers[i - 1]) ** 0.5 for i in range(1, len(self.layers))]
        self.bs = [np.random.randn(self.layers[i], 1) * (2 / self.layers[i - 1]) ** 0.5 for i in range(1, len(self.layers))]

    def fp(self, X):
        nn_out = [X]
        for i in range(len(self.layers) - 1):
            z = (nn_out[i] @ self.wgt[i].T) + self.bs[i].T
            if i == len(self.layers) - 2:
                nn_out.append(softmax(z))
            else:
                nn_out.append(self.activation(z))
        return nn_out

    def bp(self, Y, nn_out):
        delta = [(Y - nn_out[-1])]
        for i in range(len(self.layers) - 2, 0, -1):
            delta.append((delta[-1] @ self.wgt[i]) * self.activation_derivative(nn_out[i]))
        delta.reverse()
        return delta

    def fit(self, X_train, Y_train): 
        permutation = np.random.permutation(X_train.shape[0])
        X = X_train[permutation]
        Y = Y_train[permutation]
        prev_loss, curr_loss, itr = 1e9, 0, 0
        num_batches = len(X_train) / self.batch_size
        while True:
            if(abs(curr_loss - prev_loss) <= self.eps and itr >= self.max_epochs):
                break
            prev_loss = curr_loss
            for j in range(0, len(X_train), self.batch_size):
                X_batch = X[j:j + self.batch_size]
                Y_batch = Y[j:j + self.batch_size]
                nn_out = self.fp(X_batch)
                delta = self.bp(Y_batch, nn_out)
                rate = self.lr
                if self.adpt:
                    rate = self.lr / (itr+1) ** 0.5
                for i in range(len(self.layers) - 1):
                    self.wgt[i] += rate * (delta[i].T @ nn_out[i]) / self.batch_size
                    self.bs[i] += rate * np.sum(delta[i], axis=0,keepdims=True).T / self.batch_size
                curr_loss += log_loss(Y_batch, nn_out[-1])
            curr_loss = curr_loss/num_batches
            itr = itr + 1

    def predict(self, X):
        probabilities = self.fp(X)
        max_indices = np.argmax(probabilities[-1],axis=1)
        result = np.zeros_like(probabilities[-1])
        result[np.arange(X.shape[0]), max_indices] = 1
        return result
    

if __name__ == '__main__':

    x_train_path = "x_train.npy"
    y_train_path = "y_train.npy"

    X_train, y_train = get_data(x_train_path, y_train_path)

    x_test_path = "x_test.npy"
    y_test_path = "y_test.npy"

    X_test, y_test = get_data(x_test_path, y_test_path)

    label_encoder = OneHotEncoder(sparse_output = False)
    label_encoder.fit(np.expand_dims(y_train, axis = -1))

    y_train_onehot = label_encoder.transform(np.expand_dims(y_train, axis = -1))
    y_test_onehot = label_encoder.transform(np.expand_dims(y_test, axis = -1))

    # nns = [[512],[512,256],[512,256,128],[512,256,128,64]]
    nns = [[512]]
    # nns = [[10]]
    f1_scores = []
    accuracies = []
    network_depth = []
    for nn in nns:
        clf = NeuralNetwork(1024,5,nn, activation='sigmoid', 
                            learning_rate=0.01, max_epochs=500, adaptive=True, batch_size=32)
        clf.fit(X_train, y_train_onehot)
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