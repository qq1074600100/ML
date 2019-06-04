import math
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


# 使用logistic函数作为激活函数的神经网络，层数和各层节点数自定义，向量化计算
class NeuralNetworkGeneric(object):
    def __init__(self, X, Y, layerNodes, step=0.1, regLambda=0.01):
        self.layerNodes = layerNodes
        self.layers = len(layerNodes)
        self.step = step
        self.regLambda = regLambda
        self.numExamples = X.shape[0]
        self.X = X
        self.Y = Y
        self.y = np.argmax(self.Y, axis=1)

    def calLogisticFunc(self, x):
        return 1/(1+math.e**(-x))

    def calculateLoss(self, model):
        W = model
        metrixX = self.X
        metrixY = self.Y

        # 正向传播，计算预测值
        A = [metrixX]
        for i in range(self.layers-1):
            tmpAi = np.ones((A[i].shape[0], A[i].shape[1]+1), dtype='float64')
            tmpAi[:, 1:] = A[i][:, :]
            zNext = tmpAi.dot(W[i])
            aNext = self.calLogisticFunc(zNext)
            A.append(aNext)
        probs = A[-1]
        # 计算损失，向量化
        tmpMetrix = abs(metrixY-probs)
        rstMetrix = -np.log2(1-tmpMetrix)
        sumLoss = np.sum(rstMetrix)

        # 在损失上加上正则项（可选）
        itemLambda = 0
        for i in range(len(W)):
            itemLambda += np.sum(np.square(W[i][1:, :]))
        itemLambda *= self.regLambda/2
        sumLoss += itemLambda

        sumLoss /= self.numExamples
        return sumLoss

    def calModel(self):
        # 用随机值初始化参数。我们需要学习这些参数
        np.random.seed(0)
        W = []
        for i in range(self.layers-1):
            W.append(np.random.randn(
                self.layerNodes[i]+1, self.layerNodes[i+1]))

        model = W

        metrixX = self.X
        metrixY = self.Y

        lastLoss = 0
        nowLoss = self.calculateLoss(model)

        self.lossFunc = [nowLoss]

        while nowLoss-lastLoss < -0.00001 or lastLoss == 0:
            # 正向传播，计算预测值
            A = [metrixX]
            tmpA = []
            for i in range(self.layers-1):
                tmpAi = np.ones(
                    (A[i].shape[0], A[i].shape[1]+1), dtype='float64')
                tmpAi[:, 1:] = A[i][:, :]
                tmpA.append(tmpAi)
                zNext = tmpAi.dot(W[i])
                aNext = self.calLogisticFunc(zNext)
                A.append(aNext)
            probs = A[-1]

            # 反向传播
            lastDelta = probs-metrixY
            dW = [(tmpA[-1].T).dot(lastDelta)/self.numExamples]
            for i in range(self.layers-2):
                nowDelta = lastDelta.dot(W[-i-1].T) * \
                    tmpA[-i-1]*(1-tmpA[-i-1])
                tmpdW = (tmpA[-i-2].T).dot(
                    nowDelta[:, 1:].reshape(self.numExamples, W[-i-2].shape[1]))
                tmpdW /= self.numExamples
                dW.append(tmpdW)
                lastDelta = nowDelta[:, 1:].reshape(
                    (nowDelta.shape[0], nowDelta.shape[1]-1))
            dW.reverse()

            # 添加正则化项
            for i in range(len(dW)):
                dW[i][1:, :] += self.regLambda * \
                    (W[i][1:, :].reshape(W[i].shape[0]-1, W[i].shape[1]))

            # 梯度下降更新参数
            for i in range(len(W)):
                W[i] -= self.step*dW[i]

            # 为模型分配新的参数
            model = W

            lastLoss = nowLoss
            nowLoss = self.calculateLoss(model)
            self.lossFunc.append(nowLoss)

        self.resultModel = model
        plt.plot(range(len(self.lossFunc)), self.lossFunc)
        plt.show()

    def predict(self, dataX):
        W = self.resultModel
        # 正向传播，计算预测值
        A = [dataX]
        tmpA = []
        for i in range(self.layers-1):
            tmpAi = np.ones(
                (A[i].shape[0], A[i].shape[1]+1), dtype='float64')
            tmpAi[:, 1:] = A[i][:, :]
            tmpA.append(tmpAi)
            zNext = tmpAi.dot(W[i])
            aNext = self.calLogisticFunc(zNext)
            A.append(aNext)
        probs = A[-1]
        return np.argmax(probs, axis=1)
