from PIL import Image
import os
import numpy as np
import cvxopt
import cvxopt.solvers

# Define the output size after resizing
output_size = (16, 16)

# Create lists to store the resized and normalized image data and labels
data = []
labels = []

# Directory containing the training data
train_dir = os.path.join(os.getcwd(), 'svm', 'train')

print(train_dir)

# Iterate through the '0' and '1' subdirectories within the 'train' directory
for class_name in ['0', '1']:
    class_dir = os.path.join(train_dir, class_name)

    # Check if the directory exists
    if not os.path.exists(class_dir):
        print(f"Directory '{class_name}' does not exist.")
        continue

    # Iterate through image files in the directory
    for filename in os.listdir(class_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(class_dir, filename)

            # Open and resize the image
            image = Image.open(image_path)
            image = image.resize(output_size)

            # Normalize the image data by dividing by 255
            image_data = np.array(image).flatten() / 255.0

            # Append the image data to the list
            data.append(image_data)

            # Assign the label based on the directory name ('0' or '1')
            label = int(class_name)
            if(label==0):
                label=-1
            labels.append(label)

# Convert the data and labels lists to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# # Verify the shapes of the data and labels arrays
# print("Data shape:", data.shape)
# print("Labels shape:", labels.shape)

# Define the SVM training function


def train_svm(data, labels, C=1.0):
    m, n = data.shape

    # Create the necessary matrices for the quadratic programming problem
    P = cvxopt.matrix(np.outer(labels, labels) * np.dot(data, data.T))
    q = cvxopt.matrix(-np.ones(m))
    G = cvxopt.matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt.matrix(labels, (1, m), 'd')
    b = cvxopt.matrix(0.0)

    # Solve the quadratic programming problem
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Extract Lagrange multipliers (alphas)
    alphas = np.ravel(solution['x'])

    # Find support vectors (non-zero alphas)
    support_vector_indices = np.where(alphas > 1e-4)[0]

    print(f'Number of support Vectors are {len(support_vector_indices)}')
    print(f'percentage of training samples that are support Vectors are {len(support_vector_indices)*100/m}')
    # support_vectors = data[support_vector_indices]
    # support_vector_labels = labels[support_vector_indices]

    # Calculate the weight vector w
    w = np.sum((alphas[i] * labels[i] * data[i]) for i in support_vector_indices)

    # Calculate the intercept term b
    b = 0
    for i in support_vector_indices:
        b += labels[i]
        b -= np.dot(w, data[i])
    b /= len(support_vector_indices)

    return w, b

# Train the SVM
w, b = train_svm(data, labels, C=1.0)
# Now you have the weight vector w and intercept term b
# print("Weight Vector (w):", w)
# print("Intercept Term (b):", b)


#Now we need to use this model on the validation data
#These here are the validation data and validation labels
data = []
labels = []

# Directory containing the training data
train_dir = os.path.join(os.getcwd(), 'svm', 'val')

print(train_dir)

# Iterate through the '0' and '1' subdirectories within the 'train' directory
for class_name in ['0', '1']:
    class_dir = os.path.join(train_dir, class_name)

    # Check if the directory exists
    if not os.path.exists(class_dir):
        print(f"Directory '{class_name}' does not exist.")
        continue

    # Iterate through image files in the directory
    for filename in os.listdir(class_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(class_dir, filename)

            # Open and resize the image
            image = Image.open(image_path)
            image = image.resize(output_size)

            # Normalize the image data by dividing by 255
            image_data = np.array(image).flatten() / 255.0

            # Append the image data to the list
            data.append(image_data)

            # Assign the label based on the directory name ('0' or '1')
            label = int(class_name)
            if(label==0):
                label=-1
            labels.append(label)

correct=0

for i in range(len(data)):
    value = np.dot(w,data[i]) +b
    if((value<0 and labels[i]==-1) or (value>=0 and labels[i]==1)):
        correct+=1

print(f'Precentage correct is {correct*100/len(data)}')