import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sympy as sp
import datetime
from mpl_toolkits.mplot3d.axes3d import Axes3D

# 从文件中读取数据并格式化


def createDataSet(filename):
    fr = open(filename)
    lines = fr.readlines()
    m = len(lines)
    n = len(lines[0].split("\t"))
    dataSet = np.empty((m, n), dtype='float64')
    for i in range(m):
        line = lines[i]
        fields = line.split("\t")
        dataSet[i, :] = fields
    return dataSet

# 计算代价函数


def getCostFunc(dataSet):
    # 数据数
    m = dataSet.shape[0]
    # 未知参数数目
    n = dataSet.shape[1]
    costFunc = 0
    for data in dataSet:
        x = data[0:-1]
        y = data[-1]
        p0 = sp.Symbol('p'+str(0))
        # params.append((p0, 0))
        func = p0
        for i in range(n-1):
            tempP = sp.Symbol('p'+str(i+1))
            # params.append((tempP, 0))
            func = func + tempP*x[i]
        costFunc = costFunc+(func-y)**2
    costFunc = costFunc/(2*m)
    return costFunc


def gradDesc(dataSet, step, stopVal):
    n = dataSet.shape[1]
    costFunc = getCostFunc(dataSet)

    # 将代价函数对每个未知参数求偏导，返回储存所有未知参数偏导数的列表
    diffFuncs = []
    for i in range(n):
        diffFuncN = sp.diff(costFunc, 'p'+str(i))
        diffFuncs.append(diffFuncN)

    # 计算pN参数的偏导数值
    def getDiffParamN(params, n):
        return diffFuncs[n].subs(params)

    # 初始化未知参数为0，用list保存，格式为[(p0,0),(p1,0),...]
    params = []
    for i in range(n):
        params.append((sp.Symbol('p'+str(i)), 0))
    # 建立n×1矩阵，用于保存每一步计算的偏导数值
    tempDiffs = np.zeros((n), dtype='float64')
    # # 储存上一轮偏导数值
    # lastTempDiffs = np.zeros((n), dtype='float64')
    # lastTempDiffs.fill(10000)

    # 设立标记，作为循环终止条件
    sign = 0
    while True:
        # 求出各未知参数对应的偏导数值
        for i in range(n):
            tempDiffs[i] = getDiffParamN(params, i)
            # 若该值接近0，说明已接近极值点，停止梯度下降，stopVal数值极小
            if abs(tempDiffs[i]) < stopVal:
                sign = 1
                break
        if sign == 1:
            break
        # 同步更新所有未知参数值
        for i in range(n):
            params[i] = (params[i][0], params[i][1]-step*tempDiffs[i])
        # # 若本轮偏导数值绝对值大于上轮则缩短步长
        # if step > 0.0002 and np.any(abs(lastTempDiffs) < abs(tempDiffs)):
        #     step = step/2
        # lastTempDiffs[:] = tempDiffs[:]

    return params


def test(step, stopVal):
    # dataSet = np.array([[1, -1], [2, 1.5], [3, 3.3], [4, 3], [
    #     5, 4.5], [6, 5], [7, 7], [8, 7.7], [9, 8]], dtype='float64')
    # print(dataSet.dtype)
    dataSet = createDataSet(
        r"D:\computerScience\python3.7\mechanicLearning\ttt.txt")
    # step = 0.0005

    startTime = datetime.datetime.now().timestamp()
    print(startTime)

    params = gradDesc(dataSet, step, stopVal)
    print(params)

    endTime = datetime.datetime.now().timestamp()
    print(endTime)
    print("totally consume :"+str(endTime-startTime)+"s")

    fig = plt.figure()
    # ax = Axes3D(fig)
    x = np.array(dataSet[:, 0])
    y = np.array(dataSet[:, 1])
    # x, y = np.meshgrid(x, y)
    # plt.scatter(dataSet[:, 0], dataSet[:, 1])
    # plt.plot(x, y)
    # plt.title('y='+str(params[1][1])+'x+'+str(params[0][1]))
    plt.scatter(x, dataSet[:, -1])
    # ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
    x = np.arange(-5, 10, 0.1)
    z = params[1][1]*x+params[2][1]*x**2+params[3][1]*x**3+params[0][1]
    plt.plot(x, z)
    plt.show()
