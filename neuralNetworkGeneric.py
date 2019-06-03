import math
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


# 使用logistic函数作为激活函数的神经网络，三层,向量化计算
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
        # 计算损失
        sumLoss = 0
        for i in range(self.numExamples):
            for j in range(metrixY.shape[1]):
                if abs(metrixY[i, j]-1) < 0.001:
                    sumLoss -= math.log2(probs[i, j])
                elif abs(metrixY[i, j]) < 0.001:
                    sumLoss -= math.log2(1-probs[i, j])
        # 在损失上加上正则项（可选）
        # itemLambda = self.regLambda/2 *\
        #     (np.sum(np.square(W1[:, 1:])) + np.sum(np.square(W2[:, 1:])))
        # sumLoss += itemLambda
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

        while nowLoss-lastLoss < -0.000001 or lastLoss == 0:
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
            dW = [(tmpA[-1].T).dot(lastDelta)]
            for i in range(self.layers-2):
                nowDelta = lastDelta.dot(W[-i-1].T) * \
                    tmpA[-i-1]*(1-tmpA[-i-1])
                tmpdW = (tmpA[-i-2].T).dot(
                    nowDelta[:, 1:].reshape(self.numExamples, W[-i-2].shape[1]))
                tmpdW /= self.numExamples
                dW.append(tmpdW)
                lastDelta = nowDelta
            dW.reverse()
            # delta2 = (delta3.dot(W2.T))*tmpA2*(1-tmpA2)

            # dW2 = (tmpA2.T).dot(delta3)
            # dW1 = (tmpA1.T).dot(delta2[:, 1:].reshape(
            #     numExamples, self.numHide))

            # dW2 /= numExamples
            # dW1 /= numExamples

            # 正则化项
            # dW2 += self.regLambda * W2
            # dW1 += self.regLambda * W1

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
