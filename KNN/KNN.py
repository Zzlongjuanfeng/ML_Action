#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""A implementation of KNN"""
__author__ = 'xian'
from numpy import *
import operator
import numpy
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables

def classify0(inX, dataSet, labels, k):
    #compute the distance(Euclidean distance)
    dataSetSize = dataSet.shape[0]    # numpy::array.shape 返回数组的行和列
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet    #得到差值矩阵
    sqDiffMat =  diffMat ** 2    #矩阵每个元素都做平方
    sqDistances = sqDiffMat.sum(axis = 1)  #axis=1:矩阵按行加， 得到一个向量
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort() #排序并返回下标，默认升序
    #count the label of the k-nearest
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDistIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1    #如果不存在就返回0
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1), reverse = True )    #operator.itemgetter(1),按照第1维的值排序
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()    # return a list, all the lines of file in the list
    numberOfLine = len(arrayOfLines)
    returnMat = zeros((numberOfLine, 3))    # return a new array of given shape
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()  # leading and trailing whitespace removed
        listFromLine = line.split('\t') # return a list
        returnMat[index, : ] = listFromLine[0 : 3]
        classLabelVector.append(str(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingclassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix("/home/zxf/Machine_learning/dataset/"
                                              "machinelearninginaction/Ch02/datingTestSet2.txt")  #load data file
    normDataMat, ranges, minVals = autoNorm(datingDataMat)
    m = normDataMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normDataMat[i], normDataMat[numTestVecs : m, :],
                                    datingLabels[numTestVecs : m], 5)
        print("The classfier came back with: %s, the real answer: %s" %(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("the total error rate is: %f" % (errorCount / numTestVecs))
    print("total number of test samples: %d" % numTestVecs)

def classperson():
    resultList = ['not at all', 'in small dases', 'in large doses']
    gameTime = float(input("percentage of time spent playing games:"))
    flyMiles = float(input("frequent flier miles earned per year:"))
    iceCream = float(input("liters of ice cream consumed per year:"))
    inArr = array([flyMiles, gameTime, iceCream])
    datingDataMat, datingLabels = file2matrix("/home/zxf/Machine_learning/dataset/"
                                              "machinelearninginaction/Ch02/datingTestSet2.txt")  #load data file
    normDataMat, ranges, minVals = autoNorm(datingDataMat)
    classifierResult = classify0((inArr - minVals) / ranges, normDataMat, datingLabels, 5)
    print("you will probably love this person:", resultList[int(classifierResult)1 - 1])

if __name__ == '__main__':
    # group, labels = createDataSet()
    # testLabel = classify0([1, 0], group, labels, 3)
    # print(testLabel)
    # help(min)

    classperson()
    # datingclassTest()
    # fig = plt.figure()    # creates a new figure
    # ax = fig.add_subplot(111)   # add a subplot
    # ax.scatter(normDataMat[:, 0], normDataMat[:, 1], 10 * array(datingLabels), 10 * array(datingLabels))
    # plt.show()


