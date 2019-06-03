import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import neuralNetworkUtils as utils
from neuralNetworkVectorize import NeuralNetworkVectorize
from neuralNetwork import NeuralNetwork
from neuralNetworkGeneric import NeuralNetworkGeneric

# def f(x):
#     rst = x.dot(np.array([[0.3], [-1]], dtype='float64'))
#     for i in range(rst.shape[0]):
#         if rst[i, 0] > 0:
#             rst[i, 0] = 1
#         else:
#             rst[i, 0] = 0
#     return rst


X, y = utils.generate_data()
Y = np.zeros((len(y), 2), dtype='float64')
for i in range(len(y)):
    Y[i, y[i]] = 1
layerNodes = [2, 3, 3, 2]
# model = NeuralNetwork(2, 2, numHide=3, step=1, regLambda=0.01)
# model = NeuralNetworkVectorize(X, Y, numHide=3, step=1, regLambda=0.001)
model = NeuralNetworkGeneric(X, Y, layerNodes, step=1, regLambda=0.001)
model.calModel()

utils.plot_decision_boundary(lambda x: model.predict(x), X, y)
# model.plot_decision_boundary(X, y)
