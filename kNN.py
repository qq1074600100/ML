from numpy import *
import operator
import os


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqrt(sqDistances)
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    numberOfLines = len(lines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVecctor = []
    labels2num = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}
    for i in range(numberOfLines):
        line = lines[i]
        line = line.strip()
        fields = line.split('\t')
        returnMat[i, :] = fields[0:3]
        classLabelVecctor.append(labels2num[fields[-1]])
    return returnMat, classLabelVecctor


def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet-tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRadio = 0.1
    datingDataMat, datingLabels = file2matrix(
        'mechanicLearning\datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTest = int(m*hoRadio)
    errCount = 0
    for i in range(numTest):
        classifierResult = classify0(
            normMat[i, :], normMat[numTest:m, :],
            datingLabels[numTest:m], 3)
        print("result is: ", classifierResult, ", real answer is: ",
              datingLabels[i])
        if classifierResult != datingLabels[i]:
            print("false!!!")
            errCount += 1
    print("error rate is: ", (errCount/numTest))


def img2vector(filename):
    returnVector = zeros((1, 1024))
    fr = open(filename)
    lines = fr.readlines()
    for i in range(32):
        line = lines[i]
        for j in range(32):
            returnVector[0, 32*i+j] = int(line[j])
    return returnVector


def handwritingClassTest():
    hwLabels = []
    trainFileList = os.listdir(
        "D:\\computerScience\\python3.7\\mechanicLearning\\digits\\trainingDigits")
    m = len(trainFileList)
    trainMat = zeros((m, 1024))
    for i in range(m):
        filename = trainFileList[i]
        fileStr = filename.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainMat[i, :] = img2vector(
            "D:\\computerScience\\python3.7\\mechanicLearning\\digits\\trainingDigits\\"+filename)
    testFileList = os.listdir(
        "D:\\computerScience\\python3.7\\mechanicLearning\\digits\\testDigits")
    errCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        filename = testFileList[i]
        fileStr = filename.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(
            "D:\\computerScience\\python3.7\\mechanicLearning\\digits\\testDigits\\"+filename)
        classifierResult = classify0(vectorUnderTest, trainMat, hwLabels, 3)
        print("result is: ", classifierResult, ", real answer is: ",
              classNumStr)
        if classifierResult != classNumStr:
            print("false!!!")
            errCount += 1
    print("error num is: ", errCount)
    print("error rate is: ", (errCount/mTest))
