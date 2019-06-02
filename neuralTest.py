import neuralNetwork
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


# def f(x):
#     rst = x.dot(np.array([[0.3], [-1]], dtype='float64'))
#     for i in range(rst.shape[0]):
#         if rst[i, 0] > 0:
#             rst[i, 0] = 1
#         else:
#             rst[i, 0] = 0
#     return rst


model = neuralNetwork.NeuralNetwork(2, 2, numHide=3, step=1,regLambda=0.01)
model.calModel()


X, y = neuralNetwork.generate_data()
# plt.scatter(X[:, 0], X[:, 1], s=40, c=y)
neuralNetwork.plot_decision_boundary(
    lambda x: model.predict(x), X, y)
