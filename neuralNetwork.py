import math
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    # print(len(y))
    return X, y


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    temp = np.c_[xx.ravel(), yy.ravel()]
    Z = pred_func(temp.T)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

# 使用logistic函数作为激活函数的神经网络，三层


class NeuralNetwork(object):
    def __init__(self, numInput, numOutput,
                 numHide=3, step=0.01, regLambda=0.01):
        self.numInput = numInput
        self.numOutput = numOutput
        self.numHide = numHide
        self.regLambda = regLambda
        self.step = step
        self.X, self.Y = self.generate_data()

    def generate_data(self):
        np.random.seed(0)
        X, y = datasets.make_moons(200, noise=0.20)
        X = X.T
        Y = np.zeros((self.numOutput, len(y)), dtype='float64')
        for i in range(len(y)):
            Y[int(y[i]), i] = 1
        return X, Y

    def createDataSet(self, filename):
        file = open(filename)
        lines = file.readlines()
        numExamples = len(lines)
        numParams = len(lines[0].split("\t"))-1
        X = np.empty((numParams, numExamples), dtype='float64')
        Y = np.zeros((self.numOutput, numExamples), dtype='float64')
        for i in range(len(lines)):
            fields = lines[i].split("\t")
            X[:, i] = fields[:-1]
            Y[int(fields[-1]), i] = 1
        self.X = X
        self.Y = Y

    def calLogisticFunc(self, data):
        return 1/(1+math.e**(-data))

    def calculateLoss(self, model):
        W1, W2 = model['W1'], model['W2']
        metrixX = self.X
        metrixY = self.Y

        # 正向传播，计算预测值
        X = np.ones((metrixX.shape[0]+1, metrixX.shape[1]), dtype='float64')
        X[1:, :] = metrixX[:, :]
        z1 = W1.dot(X)
        tmpA1 = self.calLogisticFunc(z1)
        a1 = np.ones((tmpA1.shape[0]+1, tmpA1.shape[1]))
        a1[1:, :] = tmpA1[:, :]
        z2 = W2.dot(a1)
        probs = self.calLogisticFunc(z2)
        # 计算损失
        sumLoss = 0
        for i in range(probs.shape[1]):
            for j in range(metrixY.shape[0]):
                if abs(metrixY[j, i]-1) < 0.001:
                    sumLoss -= math.log2(probs[j, i])
                elif abs(metrixY[j, i]) < 0.001:
                    sumLoss -= math.log2(1-probs[j, i])
        # 在损失上加上正则项（可选）
        itemLambda = self.regLambda/2 *\
            (np.sum(np.square(W1[:, 1:])) + np.sum(np.square(W2[:, 1:])))
        sumLoss += itemLambda
        sumLoss /= probs.shape[1]
        return sumLoss

    def calModel(self):
        # 用随机值初始化参数。我们需要学习这些参数
        np.random.seed(0)
        W1 = np.random.randn(self.numHide, self.numInput+1)
        W2 = np.random.randn(self.numOutput, self.numHide+1)

        model = {'W1': W1, 'W2': W2}

        metrixX = self.X
        metrixY = self.Y

        lastLoss = 0
        nowLoss = self.calculateLoss(model)

        numExemples = self.Y.shape[1]
        while nowLoss-lastLoss < -0.00001 or lastLoss == 0:
            dW2 = 0
            dW1 = 0
            for i in range(numExemples):
                # 正向传播，计算预测值
                X = np.ones((metrixX.shape[0]+1, 1), dtype='float64')
                X[1:, :] = metrixX[:, i].reshape((metrixX.shape[0], 1))
                z1 = W1.dot(X)
                tmpA1 = self.calLogisticFunc(z1)
                a1 = np.ones((tmpA1.shape[0]+1, 1), dtype='float64')
                a1[1:, :] = tmpA1[:, :]
                z2 = W2.dot(a1)
                probs = self.calLogisticFunc(z2)

                # 反向传播
                delta3 = probs-metrixY[:, i].reshape((metrixY.shape[0], 1))
                delta2 = ((W2[:, 1:].T).dot(delta3))*(1 - tmpA1)*tmpA1

                # 计算偏导数

                dW2 += delta3.dot(a1[1:, :].T)
                dW1 += (delta2.reshape((self.numHide, 1))).dot(X[1:, :].T)

            dW2 += self.regLambda * \
                (W2[:, 1:].reshape((dW2.shape[0], dW2.shape[1])))
            dW1 += self.regLambda * \
                (W1[:, 1:].reshape((dW1.shape[0], dW1.shape[1])))

            dW2 /= numExemples
            dW1 /= numExemples
            # 梯度下降更新参数
            W1[:, 1:] = W1[:, 1:].reshape(self.numHide, 2) - self.step * dW1
            W2[:, 1:] = W2[:, 1:].reshape(2, self.numHide)-self.step * dW2

            # 为模型分配新的参数
            model = {'W1': W1, 'W2': W2}

            lastLoss = nowLoss
            nowLoss = self.calculateLoss(model)

        self.resultModel = model

    def predict(self, dataX):
        W1, W2 = self.resultModel['W1'], self.resultModel['W2']
        # 正向传播，计算预测值
        X = np.ones(
            (dataX.shape[0]+1, dataX.shape[1]), dtype='float64')
        X[1:, :] = dataX[:, :]
        z1 = W1.dot(X)
        tmpA1 = self.calLogisticFunc(z1)
        a1 = np.ones((tmpA1.shape[0]+1, tmpA1.shape[1]))
        a1[1:, :] = tmpA1[:, :]
        z2 = W2.dot(a1)
        probs = self.calLogisticFunc(z2)
        return np.argmax(probs, axis=0)
