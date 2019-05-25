import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def createDataSet(filename):
    fr = open(filename)
    lines = fr.readlines()
    m = len(lines)
    dataSet = np.empty((m, 2), dtype='float64')
    for i in range(m):
        line = lines[i]
        fields = line.split("\t")
        dataSet[i, :] = fields
    return dataSet


def lineFit(dataSet, step):
    m = dataSet.shape[0]

    def differentiateParam0(a, b):
        result = 0
        for data in dataSet:
            x = data[0]
            y = data[-1]
            result = result+2*(a*x+b-y)
        result = result/(2*m)
        return result

    def differentiateParam1(a, b):
        result = 0
        for data in dataSet:
            x = data[0]
            y = data[-1]
            result = result+2*(a*x+b-y)*x
        result = result/(2*m)
        return result

    p0 = 0
    p1 = 0
    diff0 = 100
    diff1 = 100
    while abs(diff0) > 0.00001 and abs(diff1) > 0.0001:
        diff0 = differentiateParam0(p1, p0)
        diff1 = differentiateParam1(p1, p0)
        p0 = p0-step*diff0
        p1 = p1-step*diff1

    return p0, p1


def test():
    # dataSet = np.array([[1, -1], [2, 1.5], [3, 3.3], [4, 3], [
    #     5, 4.5], [6, 5], [7, 7], [8, 7.7], [9, 8]], dtype='float64')
    # print(dataSet.dtype)
    dataSet = createDataSet(
        r"D:\computerScience\python3.7\mechanicLearning\ttt.txt")
    step = 0.002
    p0, p1 = lineFit(dataSet, step)
    print(p0, "\t", p1)

    fig = plt.figure()
    x = np.array(dataSet[:, 0])
    y = p1*x+p0
    plt.scatter(dataSet[:, 0], dataSet[:, 1])
    plt.plot(x, y)
    plt.title('y='+str(p1)+'x+'+str(p0))
    plt.show()
