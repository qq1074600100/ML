import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sympy as sp


class GradDesc(object):
    def __init__(self, filePath, step=0.1, paramsName=None):
        # 初始化各属性
        # 根据文件构建数据集并做特征值归一化
        self.__dataSet = self.__createDataSet(filePath)
        paramsN = self.__dataSet.shape[1]
        self.__step = step
        self.__rstParams = None
        # 若传入特征名与特征值个数不匹配则报错
        if paramsName is None:
            paramsName = [("x"+str(i), 1) for i in range(1, paramsN)]
        assert len(paramsName) == paramsN - \
            1, "The number of paramsName don't map with dataSet"
        self.__paramsName = paramsName

    def get_step(self):
        return self.__step

    def set_step(self, step):
        self.__step = step

    def set_file(self, filePath):
        self.__dataSet = self.__createDataSet(filePath)

    def get_dataSet(self):
        return self.__dataSet

    def get_paramsName(self):
        return self.__paramsName

    # 根据文件内容构建数据集
    def __createDataSet(self, filePath):
        fr = open(filePath)
        lines = fr.readlines()
        m = len(lines)
        n = len(lines[0].split("\t"))
        dataSet = np.empty((m, n), dtype='float64')
        for i in range(m):
            line = lines[i]
            fields = line.split("\t")
            dataSet[i, :] = fields
        return dataSet

    # 对数据集做归一化操作,把特征值归一化为-1~1之间，并返回归一化特征集和归一化参数，
    # 归一化参数格式为[(avg,scale),...](每个特征值对应一个tuple)
    def __normalizeDataSet(self, dataSet):
        normDataSet = np.array(dataSet, dtype='float64')
        n = normDataSet.shape[1]
        normlizeParams = []
        for i in range(n-1):
            tempCol = normDataSet[:, i]
            avg = np.average(tempCol)
            scale = (np.max(tempCol)-np.min(tempCol))/6
            normDataSet[:, i] = (tempCol-avg)/scale
            normlizeParams.append((avg, scale))
        return normDataSet, normlizeParams

    # 用梯度下降法进行计算，将计算结果保存为列表

    def __gradDescent(self, normDataSet):
        m = normDataSet.shape[0]
        n = normDataSet.shape[1]

        # 给数据矩阵最左加一列1，对应p0
        plusDataSet = np.ones((m, n+1), dtype='float64')
        plusDataSet[:, 1:] = normDataSet[:, :]

        # 取得数据矩阵中自变量部分
        metrixX = np.array(plusDataSet[:, :-1], dtype='float64')

        # 转置
        transMetrixX = metrixX.transpose()/m

        # 算出微分矩阵
        calDiffMetrix = transMetrixX.dot(plusDataSet)

        # 初始化未知参数为0，用列向量保存
        paramsCol = np.zeros((n, 1), dtype='float64')
        # 扩展params向量，増一行-1用于计算偏导数值
        tempParamsCol = np.zeros((n+1, 1), dtype='float64')

        while True:
            tempParamsCol[n, 0] = -1
            tempParamsCol[:-1, :] = paramsCol
            # 求出各未知参数对应的偏导数值的列向量
            diffCol = calDiffMetrix.dot(tempParamsCol)
            # 循环终止条件
            # if np.count_nonzero(abs(diffCol) < 0.0001) > n/2:
            if np.any(abs(diffCol) < 0.0000000001):
                break
            # 同步更新所有未知参数值
            paramsCol = paramsCol-self.__step*diffCol
            # # 若本轮偏导数值绝对值大于上轮则缩短步长
            # if step > 0.0002 and np.any(abs(lastTempDiffs) < abs(tempDiffs)):
            #     step = step/2
            # lastTempDiffs[:] = tempDiffs[:]

        # 将结果以列表形式返回
        return [x for x in paramsCol[:, 0]]

    # 返回最终结果，默认把结果转换为函数表达式返回，可重写为自己想要的输出方式
    def calModule(self):
        normDataSet, normalizeParams = self.__normalizeDataSet(
            self.__dataSet)
        rstParams = self.__gradDescent(normDataSet)

        func = rstParams[0]
        for i in range(len(self.__paramsName)):
            func = func + rstParams[i+1] * (sp.Symbol(self.__paramsName[i][0]) **
                                            self.__paramsName[i][1]-normalizeParams[i][0])/normalizeParams[i][1]
        return func

    def showResult(self):
        func = self.calModule()
        print(func)
