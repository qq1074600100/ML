from gradDescObj import GradDesc
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import sympy as sp


# 继承通用梯度下降法对象，实现曲面的拟合及可视化输出
class FaceGradDesc(GradDesc):
    def __init__(self, filePath, paramsName, step=0.1):
        super().__init__(filePath, paramsName, step=step)
        self.__paramsName = super().get_paramsName()
        self.__dataSet = super().get_dataSet()

    # 默认只输出公式，重写为输出3D曲面拟合图
    def _showResultCustom(self):
        func = super().getResultFunc()
        # 图形化界面
        fig = plt.figure()
        ax = Axes3D(fig)
        x = list(self.__dataSet[:, 0])
        y = list(self.__dataSet[:, 1])
        z = list(self.__dataSet[:, -1])
        ax.scatter(x, y, z)

        # 做X,Y轴
        scaleX = np.max(x)-np.min(x)
        scaleX = scaleX/10
        if scaleX > 1:
            scaleX = 1
        x = np.arange(np.min(x)-scaleX, np.max(x)+scaleX, scaleX)
        lenX = len(x)

        scaleY = np.max(y)-np.min(y)
        scaleY = scaleY/10
        if scaleY > 1:
            scaleY = 1
        y = np.arange(np.min(y)-scaleY, np.max(y)+scaleY, scaleY)
        lenY = len(y)

        # 构建点列表
        xy = []
        for j in range(lenY):
            for i in range(lenX):
                xy.append((x[i], y[j]))

        x, y = np.meshgrid(x, y)

        # 计算每个点对应的值
        def fxy(varXY):
            return func.subs([(sp.Symbol("x"), varXY[0]), (sp.Symbol("y"), varXY[1])])

        # 使用map()对每一个x做计算，否则会被当做矩阵进行计算，得到的f仅有一个数据，绘图失败
        f = list(map(fxy, xy))
        # 将计算结果转为lenY×lenX的矩阵,画曲面图
        f = np.array(f, dtype='float64').reshape((lenY, lenX))
        ax.plot_surface(x, y, f, rstride=1, cstride=1, cmap='rainbow')

        plt.title(func)
        plt.show()
        return func
