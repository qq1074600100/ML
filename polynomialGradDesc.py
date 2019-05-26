from gradDescObj import GradDesc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


# 继承通用梯度下降法对象，实现多项式的拟合及可视化输出
# 避免输入冗长的paramsName列表，只输入拟合最高次数即可
class PolynomialGradDesc(GradDesc):
    def __init__(self, filePath, order, step=0.1):
        tempParamsName = []
        for i in range(order):
            tempParamsName.append(("x^" + str(i+1)))

        super().__init__(filePath, tempParamsName, step=step)
        self.__dataSet = super().get_dataSet()
        assert order == self.__dataSet.shape[1]-1, "order don't map with data"

    # 默认只输出公式，重写为输出拟合图
    def _showResultCustom(self):
        func = super().getResultFunc()
        # 图形化界面
        x = list(self.__dataSet[:, 0])
        y = list(self.__dataSet[:, -1])
        plt.scatter(x, y)
        scale = np.max(x)-np.min(x)
        scale = scale/100
        if scale > 1:
            scale = 1
        x = np.arange(np.min(x)-scale, np.max(x)+scale, scale)

        def fx(x):
            return func.subs(sp.Symbol("x"), x)

        # 使用map()对每一个x做计算，否则会被当做矩阵进行计算，得到的f仅有一个数据，绘图失败
        f = list(map(fx, x))
        plt.plot(x, f)

        plt.title(func)
        plt.show()
        return func
