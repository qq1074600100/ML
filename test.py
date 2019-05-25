import kNN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import linerFit as lf
import gradientDesc as gd
import gradientDescMetrix as gdm
from figGradDesc import FigGradDesc

# group, labels = kNN.createDataSet()

# datingDataMat, datingLabels = kNN.file2matrix(
#     'mechanicLearning\datingTestSet.txt')

# dataLabel = [x*15 for x in datingLabels]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
#            dataLabel, dataLabel)
# plt.show()

# kNN.datingClassTest()

# vector = kNN.img2vector(
#     "D:\\computerScience\\python3.7\\mechanicLearning\\testDigits\\0_13.txt")
# print(vector[0, 0:32])

# kNN.handwritingClassTest()


# lf.test()


# gdm.test(0.006)

filePath = r"D:\computerScience\python3.7\mechanicLearning\ttt.txt"
gd = FigGradDesc(filePath, step=0.00005, paramsName=["x", "x^2"])
gd.calculate()
print(gd.get_paramsName())
print(gd.get_rstParams())
gd.finalModel()
