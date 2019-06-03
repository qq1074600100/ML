import math
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


# 使用logistic函数作为激活函数的神经网络，三层,向量化计算
class NeuralNetworkVectorize(object):
    def __init__(self, X, Y, numHide=3, step=0.1, regLambda=0.01):
        self.numInput = X.shape[1]
        self.numOutput = Y.shape[1]
        self.numHide = numHide
        self.step = step
        self.regLambda = regLambda
        self.X = X
        self.Y = Y
        self.y = np.argmax(self.Y, axis=1)

    def calLogisticFunc(self, x):
        return 1/(1+math.e**(-x))

    def calculateLoss(self, model):
        W1, W2 = model['W1'], model['W2']
        metrixX = self.X
        metrixY = self.Y
        y = self.y
        numExamples = metrixX.shape[0]
        numParams = metrixX.shape[1]

        # 正向传播，计算预测值
        a1 = metrixX
        tmpA1 = np.ones((numExamples, numParams+1), dtype='float64')
        tmpA1[:, 1:] = a1[:, :]
        z2 = tmpA1.dot(W1)
        a2 = self.calLogisticFunc(z2)
        tmpA2 = np.ones((numExamples, self.numHide+1), dtype='float64')
        tmpA2[:, 1:] = a2[:, :]
        z3 = tmpA2.dot(W2)
        a3 = self.calLogisticFunc(z3)
        probs = a3
        # 计算损失
        sumLoss = 0
        for i in range(numExamples):
            for j in range(metrixY.shape[1]):
                if abs(metrixY[i, j]-1) < 0.001:
                    sumLoss -= math.log2(probs[i, j])
                elif abs(metrixY[i, j]) < 0.001:
                    sumLoss -= math.log2(1-probs[i, j])
        # 在损失上加上正则项（可选）
        itemLambda = self.regLambda/2 *\
            (np.sum(np.square(W1[1:, :])) + np.sum(np.square(W2[1:, :])))
        sumLoss += itemLambda
        sumLoss /= numExamples
        return sumLoss

    def calModel(self):
        # 用随机值初始化参数。我们需要学习这些参数
        np.random.seed(0)
        W1 = np.random.randn(self.numInput+1, self.numHide)
        W2 = np.random.randn(self.numHide+1, self.numOutput)

        metrixX = self.X
        metrixY = self.Y
        numExamples = metrixX.shape[0]
        numParams = metrixX.shape[1]

        model = {'W1': W1, 'W2': W2}

        lastLoss = 0
        nowLoss = self.calculateLoss(model)

        self.lossFunc = [nowLoss]

        while nowLoss-lastLoss < -0.00001 or lastLoss == 0:
            # 正向传播，计算预测值
            a1 = metrixX
            tmpA1 = np.ones((numExamples, numParams+1), dtype='float64')
            tmpA1[:, 1:] = a1[:, :]
            z2 = tmpA1.dot(W1)
            a2 = self.calLogisticFunc(z2)
            tmpA2 = np.ones((numExamples, self.numHide+1), dtype='float64')
            tmpA2[:, 1:] = a2[:, :]
            z3 = tmpA2.dot(W2)
            a3 = self.calLogisticFunc(z3)
            probs = a3

            # 反向传播
            delta3 = probs-metrixY
            delta2 = (delta3.dot(W2.T))*tmpA2*(1-tmpA2)

            dW2 = (tmpA2.T).dot(delta3)
            dW1 = (tmpA1.T).dot(delta2[:, 1:].reshape(
                numExamples, self.numHide))

            dW2 /= numExamples
            dW1 /= numExamples

            # 正则化项
            dW2[1:, :] += self.regLambda * \
                (W2[1:, :].reshape(W2.shape[0]-1, W2.shape[1]))
            dW1[1:, :] += self.regLambda * \
                (W1[1:, :].reshape(W1.shape[0]-1, W1.shape[1]))

            # 梯度下降更新参数
            W2 = W2-self.step*dW2
            W1 = W1-self.step*dW1

            # 为模型分配新的参数
            model = {'W1': W1, 'W2': W2}

            lastLoss = nowLoss
            nowLoss = self.calculateLoss(model)
            self.lossFunc.append(nowLoss)

        self.resultModel = model
        plt.plot(range(len(self.lossFunc)), self.lossFunc)
        plt.show()

    def predict(self, dataX):
        W1, W2 = self.resultModel['W1'], self.resultModel['W2']
        numExamples = dataX.shape[0]
        numParams = dataX.shape[1]
        # 正向传播，计算预测值
        a1 = dataX
        tmpA1 = np.ones((numExamples, numParams+1), dtype='float64')
        tmpA1[:, 1:] = a1[:, :]
        z2 = tmpA1.dot(W1)
        a2 = self.calLogisticFunc(z2)
        tmpA2 = np.ones((numExamples, self.numHide+1), dtype='float64')
        tmpA2[:, 1:] = a2[:, :]
        z3 = tmpA2.dot(W2)
        a3 = self.calLogisticFunc(z3)
        probs = a3
        return np.argmax(probs, axis=1)
