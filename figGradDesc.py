from gradDescObj import GradDesc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


# 继承通用梯度下降法对象，实现结果的可视化输出
class FigGradDesc(GradDesc):
    def __init__(self, filePath, paramsName, step=0.1):
        super().__init__(filePath, paramsName, step=step)
        self.__dataSet = super().get_dataSet()

    # 默认只输出公式，可重写为自己想要的输出方式
    def _showResultCustom(self):
        func = super().get_func()
        # 图形化界面
        x = list(self.__dataSet[:, 0])
        y = list(self.__dataSet[:, -1])
        plt.scatter(x, y)
        scale = np.max(x)-np.min(x)
        scale = scale/100
        if scale > 1:
            scale = 1
        x = np.arange(np.min(x)-scale/10, np.max(x)+scale/10, scale)

        def fx(x):
            return func.subs(sp.Symbol("x"), x)

        # 使用map()对每一个x做计算，否则会被当做矩阵进行计算，得到的f仅有一个数据，绘图失败
        f = list(map(fx, x))
        plt.plot(x, f)

        plt.title(func)
        plt.show()
        return func
