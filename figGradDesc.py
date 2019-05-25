from gradDescObj import GradDesc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# 继承通用梯度下降法对象，实现结果的可视化输出
class FigGradDesc(GradDesc):
    def __init__(self, filePath, step=0.002, paramsName=None):
        super().__init__(filePath, step=step, paramsName=paramsName)
        self.__dataSet = super().get_dataSet()

    # 返回最终结果，默认把结果转换为函数表达式返回，可重写为自己想要的输出方式
    def finalModel(self):
        # 计算表达式
        rstParams = self.get_rstParams()
        paramsName = self.get_paramsName()
        func = "f = "+str(rstParams[0])
        for i in range(len(paramsName)):
            func = func+"+" + \
                str(rstParams[i+1])+"*"+paramsName[i]
        # 图形化界面
        fig = plt.figure()
        x = np.array(self.__dataSet[:, 0])
        y = np.array(self.__dataSet[:, -1])
        plt.scatter(x, y)
        scale = np.max(x)-np.min(x)
        x = np.arange(np.min(x)-scale/10, np.max(x)+scale/10, scale/10)
        f = rstParams[0]+rstParams[1]*x+rstParams[2]*x**2
        plt.plot(x, f)
        plt.title(func)
        plt.show()
