#!/usr/bin/python3
# -*- coding: utf-8 -*-
from numpy import *
import operator

def  createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables

def  classify0(inX, dataSet, lables, k):
    dataSetSize = dataSet.shape[0]
    return dataSetSize

group, lables = createDataSet()
print("")