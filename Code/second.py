import numpy as np
import random
import matplotlib.pyplot as plt

# functions

def max_index(m):
    tmp = None
    index = None

    for i in range(m.size):
        if tmp is None:
            tmp = m[i][0]
            index = i

        elif m[i][0] > tmp:
            tmp = m[i][0]
            index = i

    return index

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))



# *************************************** Begining of first part ***************************************


# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


# Reading The Train Set
train_images_file = open('train-images.idx3-ubyte', 'rb')
train_images_file.seek(4)
num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
train_images_file.seek(16)

train_labels_file = open('train-labels.idx1-ubyte', 'rb')
train_labels_file.seek(8)

train_set = []
num_of_train_images = 100
for n in range(num_of_train_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

    label_value = int.from_bytes(train_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1
    
    train_set.append((image, label))


# # Reading The Test Set
# test_images_file = open('t10k-images.idx3-ubyte', 'rb')
# test_images_file.seek(4)

# test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
# test_labels_file.seek(8)

# num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
# test_images_file.seek(16)

# test_set = []
# for n in range(num_of_test_images):
#     image = np.zeros((784, 1))
#     for i in range(784):
#         image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256
    
#     label_value = int.from_bytes(test_labels_file.read(1), 'big')
#     label = np.zeros((10, 1))
#     label[label_value, 0] = 1
    
#     test_set.append((image, label))


# # # Plotting an image
# # show_image(train_set[0][0])
# # plt.show()


# *************************************** End of first part ***************************************

# *************************************** Begining of second part ***************************************

# random weights and biases
weight1 = np.random.randn(16,784)
weight2 = np.random.randn(16,16)
weight3 = np.random.randn(10,16)
bias1 = np.zeros((16,1))
bias2 = np.zeros((16,1))
bias3 = np.zeros((10,1))

# Accuracy calculator
count = 0
for i in range(100):

    a1 = sigmoid((weight1 @ train_set[i][0]) + bias1)
    a2 = sigmoid((weight2 @ a1) + bias2)
    a3 = sigmoid((weight3 @ a2) + bias3)
    
    guess = max_index(a3)

    if  train_set[i][1][guess] == 1:
        count += 1
    
print(" Accuracy: " + str(count) )

# *************************************** End of second part ***************************************
