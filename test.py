import kNN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import linerFit as lf
import gradientDesc as gd
import gradientDescMetrix as gdm
from gradDescObj import GradDesc
from figGradDesc import FigGradDesc
from polynomialGradDesc import PolynomialGradDesc
from faceGradDesc import FaceGradDesc
from logisticRegression import LogisticRegression

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
gd = LogisticRegression(
    filePath, ["x^1", "y^1"], step=1)
# gd = PolynomialGradDesc(filePath, 3, step=0.2)
gd.showResult()
print(gd.predict([3, 3]))
# print("f = ", func)
# print(func.subs(sp.Symbol("x"), 0))
