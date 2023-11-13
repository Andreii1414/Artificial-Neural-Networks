import numpy as np
import requests
import pickle, gzip, numpy
import matplotlib.pyplot as plt

url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
with open("mnist.pkl.gz", "wb") as fd:
    fd.write(requests.get(url).content)

with gzip.open("mnist.pkl.gz", "rb") as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding="latin")

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_x = np.concatenate((train_x, test_x), axis=0)
train_y = np.concatenate((train_y, test_y), axis=0)


def train_perceptron(train_x, train_y):
    lr = 0.01
    nr_samples, nr_features = train_x.shape
    w = np.zeros(nr_features + 1)

    for i in range(10):
        for i in range(nr_samples):
            x = train_x[i] #input
            y = train_y[i] #label
            classified = np.dot(x, w[1:]) + w[0]

            if y * classified <= 0:
                w[1:] += lr * y * x  #learning rate * label * input
                w[0] += lr * y
    return w

perceptrons = {}
for i in range(10):
    y = np.where(train_y == i, 1, -1)
    weights = train_perceptron(train_x, y)
    perceptrons[i] = weights

def predict_digit(image_input):
    max_activation = -np.inf
    prediction = None
    for digit, weights in perceptrons.items():
        image_input_c = np.concatenate(([1], image_input))
        activation = np.dot(image_input_c, weights)
        if activation > max_activation:
            max_activation = activation
            prediction = digit

    return prediction

def calculate_accuracy(data_x, data_y):
    correct = 0
    for i in range(len(data_x)):
        prediction = predict_digit(data_x[i])
        if prediction == data_y[i]:
            correct += 1
    return correct / len(data_x)

print("Acuratete pe setul de date de antrenament: ", calculate_accuracy(train_x, train_y))
print("Acuratete pe setul de date de validare: ", calculate_accuracy(valid_x, valid_y))
