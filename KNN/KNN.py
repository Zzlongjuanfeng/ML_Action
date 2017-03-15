#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""a implementation of KNN"""
__author__ = 'xian'
from numpy import *
import numpy
import operator

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

if __name__ == '__main__':
    group, labels = createDataSet()
    testLabel = classify0([1, 0], group, labels, 3)
    print(testLabel)
    # help(operator.itemgetter)
