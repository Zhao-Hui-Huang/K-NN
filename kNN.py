from numpy import *
import operator
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classcount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classcount[voteIlabel] = classcount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingDataLabels = file2matrix('datingTestSet2.txt')
    normMat = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingDataLabels[numTestVecs:m], 3)
        print("predict:{},real:{}".format(classifierResult, datingDataLabels[i]))
        if classifierResult != datingDataLabels[i]:
            errorCount += 1.0
    print("error count:{},all count :{}".format(errorCount, float(numTestVecs)))
    print("error rate is :{}".format(errorCount / float(numTestVecs)))


def file2matrix(filename):
    """
    将文本记录转为数组
    :param filename:
    :return:
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    归一化特征
    :param dataSet:
    :return:
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    range = maxVals - minVals;
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(range, (m, 1))
    return normDataSet


def main():
    datingClassTest()


if __name__ == '__main__':
    main()
