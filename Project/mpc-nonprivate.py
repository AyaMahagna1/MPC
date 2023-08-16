#!/usr/bin/env python
# coding: utf-8
import time

import numpy as np


class DO:
    data = []

    def __init__(self, vecLength, numOfRows):
        self.vecLength = vecLength
        self.numberOfRows = numOfRows
        self.generateDataSet()

    # we used random data to be our dataset
    # vecLength is the number of the features
    def generateDataSet(self):
        for i in range(self.numberOfRows):
            features = (np.random.rand(self.vecLength))
            y = (np.random.rand())
            self.data.append([features, y])

    # returning the data set
    def getDataset(self):
        return self.data


class MLE:
    # we assumed that the number of rows of any DO is the same, but even if it's not the same we'll have the same result
    def __init__(self, parties, numOfRows):
        self.parties = parties
        self.numOfRows = numOfRows

    def returnWStar(self):
        A = []
        b = []
        for i in range(len(self.parties)):
            for j in range(self.numOfRows):
                A.append(self.parties[i].getDataset()[j][0])
                b.append(self.parties[i].getDataset()[j][1])

        # this fun is using the 2-norm to solve the equation
        wStar = np.linalg.lstsq(np.array(A, dtype=np.float), np.array(b, dtype=np.float), rcond=None)[0]
        return wStar


m = 1  # number of DOs
n = 10  # num of features
d = 10  # num of rows

parties = []
t0 = time.time()
for i in range(m):
    DOi = DO(n, d)
    parties.append(DOi)
print("n=", n, "d=", d)
MLE1 = MLE(parties, d)
print("Execution Time:", time.time() - t0, " seconds")
print("w*= ", MLE1.returnWStar())
