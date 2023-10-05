import os
from PIL import Image
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import json

def gaussian_kernel(x, y, gamma):
    distance = np.linalg.norm(x - y)
    kernel_value = np.exp(-distance**2 * gamma)
    return kernel_value

def gaussian_kernel_matrix(data, labels, gamma):
    num_samples = len(data)
    kernel_matrix = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(num_samples):
            kernel_matrix[i, j] = gaussian_kernel(data[i], data[j], gamma)*labels[i]*labels[j]

    return kernel_matrix

def cvxopt_svm(examples, labels):
    C = 1.0
    m,_ = examples.shape
    X_dash = examples * labels[:, np.newaxis]
    H = gaussian_kernel_matrix(examples, labels, 0.001)
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(labels.reshape(1,-1))
    b = cvxopt_matrix(np.zeros(1))

    cvxopt_solvers.options['show_progress'] = False
    return cvxopt_solvers.qp(P, q, G, h, A, b)


if __name__ == '_main_':

    train_folders = [('train/3',-1),('train/4',1)]
    test_folders = [('val/3',-1),('val/4',1)]

    target_size = (16, 16)
    examples = np.empty((0, 768))
    test_set = np.empty((0, 768))
    labels_examples = np.empty((0,))
    labels_test = np.empty((0,))


    for folder in train_folders:
        input_folder, label = folder
        for filename in os.listdir(input_folder):
            with Image.open(os.path.join(input_folder, filename)) as img:
                img = img.resize(target_size)  
                img_arr = np.array(img)
                flattened_vector = img_arr.reshape(768)
                examples = np.vstack((examples,flattened_vector))
                labels_examples = np.append(labels_examples,label)
    
    for folder in test_folders:
        input_folder, label = folder
        for filename in os.listdir(input_folder):
            with Image.open(os.path.join(input_folder, filename)) as img:
                img = img.resize(target_size)  
                img_arr = np.array(img)
                flattened_vector = img_arr.reshape(768)
                test_set = np.vstack((test_set,flattened_vector))
                labels_test = np.append(labels_test,label)

    data = {
        'examples': examples.tolist(),
        'labels_example': labels_examples.tolist(),
        'test_set': test_set.tolist(),
        'labels_test': labels_test.tolist()
    }

    data = {}

    with open('persistent_variables.json', 'r') as file:
        data = json.load(file)

    examples = np.array(data['examples'])
    test_set = np.array(data['test_set'])
    labels_examples = np.array(data['labels_example'])
    labels_test = np.array(data['labels_test'])

    examples /= 255
    test_set /= 255

    sol = cvxopt_svm(examples,labels_examples)
    alphas = np.array(sol['x'])
    indices = np.where(alphas>1e-5)[0]
    print('Number of support vectors', len(indices))
    print('Percentage of support vectors', (len(indices)/len(examples))*100)

    kernel_matix = gaussian_kernel_matrix(examples, labels_examples, 0.001)

    b = 0
    for i in indices:
        b += labels_examples[i]
        for j in indices:
            b -= alphas[j]*labels_examples[j]*kernel_matix[i,j]
    b /= len(indices)
    print('b = ', b)

    cnt = 0
    for i in range(len(test_set)):
        y = b
        for j in indices:
            y += alphas[j]*labels_examples[j]*gaussian_kernel(examples[j], test_set[i], 0.001)
        if np.sign(y)==labels_test[i]:
            cnt+=1

    print("Accuracy", (cnt/len(test_set))*100)
    