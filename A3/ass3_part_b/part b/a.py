import numpy as np 
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def get_data(x_path, y_path):
    x = np.load(x_path)
    y = np.load(y_path)
    y = y.astype('float')
    x = x.astype('float')
    x = 2*(0.5 - x/255)
    return x, y

def get_metric(y_true, y_pred):
    results = classification_report(y_true, y_pred)
    print(results)
    avg_score = f1_score(y_true,y_pred,average='macro')
    print(avg_score)
    return avg_score

def softmax(data):
    max_vals = np.max(data, axis=1, keepdims=True)
    exp_data = np.exp(data - max_vals)
    softmax_data = exp_data / np.sum(exp_data, axis=1, keepdims=True)
    return softmax_data

class NeuralNetwork:
    def __init__(self, X, Y, layers, activation, activation_derivative, learning_rate=0.1, threshold=1e-8, mini_batch_size=100, epochs=1000, adaptive=False):
        self.X, self. Y = X, Y
        self.layers = [X.shape[1]] + layers + [Y.shape[1]]
        self.weights, self.biases = [], []
        self.activation, self.activation_derivative = activation, activation_derivative
        self.learning_rate = learning_rate if not adaptive else learning_rate * 10
        self.threshold = threshold
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.adaptive = adaptive
        for i in range(1, len(self.layers)):
            input_units, output_units = self.layers[i - 1], self.layers[i]
            scale = np.sqrt(2.0 / input_units)
            self.weights.append(np.random.randn(output_units, input_units) * scale)
            self.biases.append(np.random.randn(output_units, 1) * scale)
        
    def forward(self, X):
        self.a = [X]
        for i in range(len(self.layers) - 1):
            z = np.dot(self.a[i], self.weights[i].T) + self.biases[i].T
            activation_func = softmax if i == len(self.layers) - 2 else self.activation
            self.a.append(activation_func(z))
        return self.a[-1]

    def backward(self, Y):
        error = Y - self.a[-1]
        sigmoid_derivative = lambda z: z * (1 - z)
        self.delta = [error * sigmoid_derivative(self.a[-1])]
        self.delta += [
            self.delta[-1].dot(self.weights[i]) * self.activation_derivative(self.a[i])
            for i in range(len(self.layers) - 2, 0, -1)
        ]
        self.delta.reverse()


    def train(self):
        # Inlined Lambda to update the parameters faster.
        update_params = lambda weights, biases, delta, a, mini_batch_size, learning_rate: (
            weights + learning_rate * np.matmul(delta.T, a) / mini_batch_size,
            biases + learning_rate * np.sum(delta, axis=0, keepdims=True).T / mini_batch_size
        )

        pcost = 1000000007
        cost = 0
        for i in range(self.epochs):
            
            if (abs(cost - pcost) < self.threshold):
                break
            
            # Permute the data at every epoch
            permutation = np.random.permutation(self.X.shape[0])
            self.X = self.X[permutation]
            self.Y = self.Y[permutation]
            
            # Update the previous cost
            pcost = cost
            
            # Adaptive Learning
            rate = self.learning_rate / ((i + 1) ** 0.5) if self.adaptive else self.learning_rate
            
            # Gradient Descent 
            sz = 0
            for j in range(0, len(self.X), self.mini_batch_size):
                
                # Generating Mini Batch
                X = self.X[j:j + self.mini_batch_size]
                Y = self.Y[j:j + self.mini_batch_size]
                output = self.forward(X)
                self.backward(Y)
                
                for k in range(len(self.layers) - 1):
                    self.weights[k], self.biases[k] = update_params(
                        self.weights[k], self.biases[k], self.delta[k], self.a[k], self.mini_batch_size, rate
                    )
                
                cost += log_loss(Y, output)
                
                sz += 1
                
            cost /= sz

    def predict(self, X):
        probabilities = self.forward(X)
        result = np.zeros_like(probabilities)
        result[np.arange(X.shape[0]), np.argmax(probabilities, axis=1)] = 1
        return result
