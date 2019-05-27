import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sympy as sp
import math

# 逻辑回归基类，可以由该类派生其他类处理特定表达式的拟合
# 接受数据文件路径，参数名列表，及学习速率(默认0.1)
# 对外方法：
#   calModule():内部调用具体算法，并将返回结果转化为表达式字符串
#   showResult():若传入数据，尚未进行计算，则调用计算过程，将计算过程分析显示，并将结果按自定义方式返回，默认为表达式字符串
#   _showResultCustom():模板方法模式，该方法可被重写为自己想要的显示方式
#   getResultFunc():返回可以计算的表达式对象，而不是字符串
#   （注意：该方法只对幂函数表达式有效，涉及指数或三角函数等时无效，调用会出错！！！）
#   predict(x):预测传入数据的结果


class LogisticRegression(object):
    def __init__(self, filePath, paramsName, step=0.1):
        # 初始化各属性
        # 根据文件构建数据集并做特征值归一化
        self.__dataSet = self.__createDataSet(filePath)
        paramsN = self.__dataSet.shape[1]
        self.__step = step
        self.__rstParams = None
        # 若传入特征名与特征值个数不匹配则报错
        assert len(paramsName) == paramsN - \
            1, "The number of paramsName don't map with dataSet"
        self.__paramsName = paramsName
        self.__JFuncs = []
        # 标记位，标记最新数据集和参数条件下，是否进行过学习过程
        self.__hasCal = False

    def get_step(self):
        return self.__step

    def set_step(self, step):
        self.__step = step
        self.__hasCal = False

    def set_file(self, filePath):
        self.__dataSet = self.__createDataSet(filePath)
        self.__hasCal = False

    def get_dataSet(self):
        return self.__dataSet

    def get_paramsName(self):
        return self.__paramsName

    def get_JFuncs(self):
        return self.__JFuncs

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
        normalizeParams = []
        for i in range(n-1):
            tempCol = normDataSet[:, i]
            avg = np.average(tempCol)
            scale = (np.max(tempCol)-np.min(tempCol))/6
            normDataSet[:, i] = (tempCol-avg)/scale
            normalizeParams.append((avg, scale))
        self.__normDataSet = normDataSet
        self.__normalizeParams = normalizeParams
        return normDataSet, normalizeParams

    # 用梯度下降法进行计算，将计算结果保存为列表

    def __gradDescent(self):
        normDataSet, normalizeParams = self.__normalizeDataSet(
            self.__dataSet)

        m = normDataSet.shape[0]
        n = normDataSet.shape[1]

        # 给数据矩阵最左加一列1，对应p0
        plusDataSet = np.ones((m, n+1), dtype='float64')
        plusDataSet[:, 1:] = normDataSet[:, :]

        # 取得数据矩阵中自变量部分X和结果y向量
        metrixX = np.array(plusDataSet[:, :-1], dtype='float64')
        yCol = np.array(plusDataSet[:, -1], dtype='float64')
        yCol = yCol.reshape((m, 1))

        # 转置X
        transMetrixX = metrixX.transpose()/m

        # # 算出微分矩阵
        # calDiffMetrix = transMetrixX.dot(plusDataSet)

        # 初始化未知参数为0，用列向量保存
        paramsCol = np.ones((n, 1), dtype='float64')

        def calA(metrixX, paramsCol):
            return metrixX.dot(paramsCol)

        def logisticFunc(z):
            result = (1+math.e**(-z))**(-1)
            return result

        def hypFunc(xRow, paramsCol):
            linerResult = xRow.dot(paramsCol)
            linerResult = linerResult[0]
            result = logisticFunc(linerResult)
            return result

        def calJFunc(metrixX, yCol, nowParamsCol):
            result = 0
            m = metrixX.shape[0]
            for i in range(m):
                xRow = metrixX[i, :]
                y = yCol[i]
                if y == 0:
                    result = result+math.log2(1-hypFunc(xRow, nowParamsCol))/m
                elif y == 1:
                    result = result+math.log2(hypFunc(xRow, nowParamsCol))/m
            result = result*(-1)
            return result

        lastJFunc = 0

        countOfWhile = 0
        while True:
            countOfWhile = countOfWhile+1
            # 循环终止条件
            JFunc = calJFunc(metrixX, yCol, paramsCol)
            self.__JFuncs.append([countOfWhile, JFunc])
            if JFunc - lastJFunc > -0.000001 and lastJFunc != 0:
                break
            lastJFunc = JFunc
            # 同步更新所有未知参数值
            A = calA(metrixX, paramsCol)
            gA = logisticFunc(A)
            diffCol = transMetrixX.dot(gA-yCol)
            paramsCol = paramsCol-self.__step*diffCol
            # # 若本轮偏导数值绝对值大于上轮则缩短步长
            # if step > 0.0002 and np.any(abs(lastTempDiffs) < abs(tempDiffs)):
            #     step = step/2
            # lastTempDiffs[:] = tempDiffs[:]

        paramsRow = paramsCol.reshape((n))
        tempRstParams = [x for x in paramsRow[:]]
        for i in range(1, len(tempRstParams)):
            tempRstParams[i] = tempRstParams[i]/normalizeParams[i-1][1]
            tempRstParams[0] = float(tempRstParams[0]) - \
                float(tempRstParams[i]) * float(normalizeParams[i-1][0])

        self.__rstParams = tempRstParams

    # 返回最终结果，默认把结果转换为函数表达式返回，可重写为自己想要的输出方式
    def calModule(self):
        self.__gradDescent()
        rstParams = self.__rstParams
        module = str(float(rstParams[0]))
        for i in range(len(self.__paramsName)):
            module = module + "+" + \
                "("+str(float(rstParams[i+1]))+")" + "*" + self.__paramsName[i]

        module = "f=1/(1+e^-("+module+"))"

        self.__module = module
        # 标记已进行过学习过程
        self.__hasCal = True

    # 模板方法，先做学习过程分析，再根据自定义的方式输出结果
    def showResult(self):
        # 若尚未计算或数据有变，则需要重新计算
        if not self.__hasCal:
            self.calModule()
        # 显示结果
        self.__showJFuncChange()
        self._showResultCustom()

    # 学习过程分析
    def __showJFuncChange(self):
        # 对学习算法过程中costFunction的变化做分析
        tempJFuncs = np.array(self.__JFuncs, dtype='float64')
        x = tempJFuncs[:, 0]
        y = tempJFuncs[:, 1]
        plt.plot(x, y)
        plt.title("change of JFunc\ntotally consume " +
                  str(len(self.__JFuncs))+" circles")
        plt.show()

    # 自定义数据显示方式，默认只输出结果公式，可重写为想要的方式
    def _showResultCustom(self):
        # 输出结果公式
        module = self.__module
        print(module)

    def predict(self, x):
        result = self.__rstParams[0]
        for i in range(len(x)):
            result += x[i]*self.__rstParams[i+1]
        result = 1/(1+math.e**(-result))
        return result
