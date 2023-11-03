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
    
def get_score(network, X_test, y_test_onehot):
    predictions = network.predict(X_test)
    f1, acc = get_metric(y_test_onehot, predictions), accuracy_score(y_test_onehot, predictions)
    return f1, acc


def plot_metrics(depths, f1_scores, accuracies, filename):
    plt.figure()
    
    # Plot F1 Scores
    plt.subplot(121)
    plt.plot(depths, f1_scores)
    plt.xlabel("Number of Hidden Layer Units")
    plt.ylabel("F1 Score")
    
    # Plot Accuracies
    plt.subplot(122)
    plt.plot(depths, accuracies)
    plt.xlabel("Number of Hidden Layer Depth")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(filename)


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
    
    # networks = [512, 256, 128, 64]
    # networks = [networks[0:i + 1] for i in range(4)]4
    networks = [[1],[5],[10],[50],[100]]
    networks = [NeuralNetwork(X = X_train, Y = y_train_onehot, layers = network, activation = lambda z: 1 / (1 + np.exp(-z)), activation_derivative = lambda z: z * (1 - z), mini_batch_size=32, learning_rate=0.01, epochs=500, adaptive=True) for network in networks]
    f1_scores, accuracies = [], []
    for network in networks:
        network.train()
        f1, acc = get_score(network, X_test, y_test_onehot)
        f1_scores.append(f1)
        accuracies.append(acc)
        
    plot_metrics([1, 5, 10, 50, 100], f1_scores, accuracies, "b.png")