import numpy as np 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
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
    return f1_score(y_true,y_pred,average='macro')


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

    networks = [512, 256, 128, 64]
    networks = [networks[0:i + 1] for i in range(4)]
    f1_scores = []
    accuracies, network_depth = [], []
    for network in networks:
        model = MLPClassifier(hidden_layer_sizes=network, max_iter=500, solver='sgd', alpha=0, batch_size=32, learning_rate = 'invscaling')
        model.fit(X_train, y_train_onehot)
        y_pred = model.predict(X_test)
        f1_scores.append(get_metric(y_test_onehot,y_pred))
        accuracies.append(accuracy_score(y_test_onehot,y_pred))

    plt.figure(1)
    plt.plot(network_depth,f1_scores)
    plt.xlabel("Network Depth")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Network Depth")
    plt.savefig('f_1')    

    plt.figure(2)
    plt.plot(network_depth,accuracies)
    plt.xlabel("Network Depth")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Network Depth")
    plt.savefig('f_2') 

    