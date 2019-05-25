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


def gradDesc(dataSet, step):
    m = dataSet.shape[0]
    n = dataSet.shape[1]

    # 给数据矩阵最左加一列1，对应p0
    plusDataSet = np.ones((m, n+1), dtype='float64')
    plusDataSet[:, 1:] = dataSet[:, :]

    # 取得数据矩阵中自变量部分
    metrixX = np.array(plusDataSet[:, :-1])

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
        if np.any(abs(diffCol) < 0.0001):
            break
        # 同步更新所有未知参数值
        paramsCol = paramsCol-step*diffCol
        # # 若本轮偏导数值绝对值大于上轮则缩短步长
        # if step > 0.0002 and np.any(abs(lastTempDiffs) < abs(tempDiffs)):
        #     step = step/2
        # lastTempDiffs[:] = tempDiffs[:]

    return paramsCol


def test(step):
    # dataSet = np.array([[1, -1], [2, 1.5], [3, 3.3], [4, 3], [
    #     5, 4.5], [6, 5], [7, 7], [8, 7.7], [9, 8]], dtype='float64')
    # print(dataSet.dtype)
    dataSet = createDataSet(
        r"D:\computerScience\python3.7\mechanicLearning\ttt.txt")
    # step = 0.0005

    startTime = datetime.datetime.now().timestamp()
    print(startTime)

    paramsCol = gradDesc(dataSet, step)
    print(paramsCol)

    endTime = datetime.datetime.now().timestamp()
    print(endTime)
    print("totally consume :"+str(endTime-startTime)+"s")

    fig = plt.figure()
    # ax = Axes3D(fig)
    x = np.array(dataSet[:, 0])
    y = np.array(dataSet[:, -1])
    # x, y = np.meshgrid(x, y)
    plt.scatter(x, y)
    # ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
    x = np.arange(np.min(x)-10, np.max(x)+10, 0.5)
    fX = paramsCol[1][0]*x+paramsCol[0][0]
    plt.plot(x, fX)
    plt.show()
