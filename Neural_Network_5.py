import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork:
    layers = []
    def __init__(self, n_inputs, n_neurons, operation):
        self.weight = np.random.rand(n_neurons, n_inputs) - 0.5
        self.bias = np.zeros((n_neurons,1))
        self.operation = operation
        NeuralNetwork.layers.append(self)

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def softmax(self, Z):
        Z = Z - np.max(Z, axis=0, keepdims=True)
        return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

    def forward(self, x):
        self.output = np.dot(self.weight, x) + self.bias
        if self.operation == 'sigmoid':
            self.func_operation = self.sigmoid(self.output)

        elif self.operation == 'ReLU':
            self.func_operation = self.ReLU(self.output)

        elif self.operation == 'softmax':
            self.func_operation = self.softmax(self.output)

class Backward_prop(NeuralNetwork):
    def __init__(self, layers, outputs):
        self.layers = layers
        self.outputs = outputs

    def one_hot_encoding(self, Y):
        return np.eye(self.outputs)[Y].T

    def derivative(self, layer):
        if layer.operation == 'sigmoid':
            return layer.func_operation * (1 - layer.func_operation)

        elif layer.operation == 'ReLU':
            return layer.output > 0

    def update_params(self, learning_rate):
        for i in self.layers:
            i.weight -= learning_rate * i.dW
            i.bias -= learning_rate * i.db

    def backward_prop(self, X, Y, learning_rate=0.1):
        m = Y.size
        one_hot_Y = self.one_hot_encoding(Y)
        self.layers[-1].dZ = 2 * (self.layers[-1].func_operation - one_hot_Y)
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].dW = 1 / m * np.dot(self.layers[i].dZ, self.layers[i-1].func_operation.T)
            self.layers[i].db = np.mean(self.layers[i].dZ, axis=1).reshape(-1,1)
            self.layers[i-1].dZ = (np.dot(self.layers[i].weight.T, self.layers[i].dZ) *
                                   self.derivative(self.layers[i-1]))

        self.layers[0].dW = 1 / m * np.dot(self.layers[0].dZ, X.T)
        self.layers[0].db = np.mean(self.layers[0].dZ, axis=1).reshape(-1,1)
        self.update_params(learning_rate)


class Data_analysis:

    def predict(self, Z):
        return np.argmax(Z, 0)

    def accuracy(self, y_pred, y_true):
        return np.sum(y_pred == y_true) / y_true.size

data = pd.read_csv('datasets/training/train.csv')
data = np.array(data)
m ,n = data.shape
np.random.shuffle(data)

data_train = data[1000:m].T
X_train = data_train[1:n]
y_train = data_train[0]
X_train = X_train / 255.0

data_test = data[0:1000].T
X_test = data_test[1:n]
y_test = data_test[0]
X_test = X_test / 255.0

layer1 = NeuralNetwork(784,10,'ReLU')
layer2 = NeuralNetwork(10,10,'ReLU')
layer3 = NeuralNetwork(10,10,'softmax')
backprop = Backward_prop((layer1, layer2, layer3), 10)
data_analysis = Data_analysis()

history = []
for i in range(500):
    layer1.forward(X_train)
    layer2.forward(layer1.func_operation)
    layer3.forward(layer2.func_operation)
    backprop.backward_prop(X_train, y_train, 0.1)
    history.append(data_analysis.accuracy(data_analysis.predict(layer3.func_operation), y_train))
    if i % 10 == 0:
        print('iteration: ', i)
        print(data_analysis.predict(layer3.func_operation), y_train, end=' ')
        print('accuracy: ', data_analysis.accuracy(data_analysis.predict(layer3.func_operation), y_train))


plt.title('NeuralNet')
plt.plot(np.arange(len(history)),history)
plt.xlabel('iterations')
plt.ylabel('accuracy')
layer1.forward(X_test)
layer2.forward(layer1.func_operation)
layer3.forward(layer2.func_operation)
print('Final accuracy: ', data_analysis.accuracy(data_analysis.predict(layer3.func_operation), y_test) * 100)
plt.show()