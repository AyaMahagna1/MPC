#!/usr/bin/env python
# coding: utf-8
import hashlib
import pickle
import queue
import random
import threading
import time

import numpy as np
from phe import paillier


# implementation of ciphertext for labHE
class labeled_ciphertext:
    def __init__(self):
        pass

    def __add__(self, other):
        ret = labHE(self.pk)
        ret.c = self.c + other.c
        ret.a = self.a + other.a
        return ret

    def __mul__(self, other):
        return self.pk.encrypt(self.a * other.a) + self.c * other.a + other.c * self.a


class labHE:
    # localGen
    def __init__(self, pk, F, Lambda=128):
        self.pk = pk
        self.F = F
        self.sigma = random.randint(0, (1 << Lambda) - 1)
        self.pk2 = pk.encrypt(self.sigma)

    # create a labeled_ciphertext from msg and label
    def labEnc(self, msg, tau):
        b = self.F(self.sigma, tau)
        v = float(b / 2. ** 253)
        c = self.pk.encrypt(v)
        lc = labeled_ciphertext()
        lc.a = msg - float(v)
        lc.c = c
        lc.pk = self.pk
        return lc

    # mulList is made from ((owner_number1,label1),(owner_number2,label2))
    def labDec(self, sk, pkList, mulList, cTilde):
        sigmaList = [sk.decrypt(x) for x in pkList]
        bTilde = sum(
            [self.F(sigmaList[pair[0][0]], pair[0][1]) * self.F(sigmaList[pair[1][0]], pair[1][1]) for pair in mulList])
        mTilde = sk.decrypt(cTilde) + bTilde
        return mTilde


# we implement F1 based on hash libs' sha3 functions
def F1(sigma, label, Lambda=128):
    s = hashlib.sha3_256(
        sigma.to_bytes(Lambda, "little") + label[0].to_bytes(256, "little") + label[1].to_bytes(256, "little")).digest()
    return int.from_bytes(s, "little")


class CSP:
    def __init__(self, qInDO, qOutDO, qInMLE, qOutMLE, n, d):
        self.qInDO = qInDO
        self.qOutDO = qOutDO
        self.qInMLE = qInMLE
        self.qOutMLE = qOutMLE
        self.n = n
        self.d = d

    def setPartition(self, partition):
        self.partition = partition

    def Run(self):
        self.phase1()
        self.phase2()

    def phase1(self):
        # (step1:key-generation) genrating the pair of keys and distribute it
        (pk, sk) = paillier.generate_paillier_keypair()
        self.sk = sk
        self.pk = pk
        for qOut in self.qOutDO:
            qOut.put(pk)
        self.qOutMLE.put(pk)
        X = []
        for _ in range(self.d):
            X.append([None] * (self.n + 1))
        for i in range(len(self.partition)):
            do_k = self.partition[i]
            pk_k = self.qInDO[i].get()
            sigma_k = sk.decrypt(pk_k)
            for i, j in do_k:
                X[i - 1][j - 1] = float(F1(sigma_k, (i, j)) / 2. ** 253)
        A = []
        for _ in range(self.n):
            rowA = []
            for _ in range(self.n):
                rowA.append(pk.encrypt(0))
            A.append(rowA)
        b = []
        for _ in range(self.n):
            b.append(pk.encrypt(0))
        for i in range(self.n):
            for j in range(self.n):
                for t in range(self.d):
                    A[i][j] = A[i][j] + X[t][i] * X[t][j]
        for i in range(self.n):
            for t in range(self.d):
                b[i] = b[i] + X[t][i] * X[t][-1]
        self.qOutMLE.put((A, b))

    def phase2(self):
        # Step 2 : masked model computation
        masked_data = self.qInMLE.get()
        solution = self.solve(masked_data)
        self.qOutMLE.put(solution)

    def solve(self, data):
        # decoding
        A = []
        for i in range(len(data[0])):
            row = []
            for j in range(len(data[0][0])):
                row.append(self.sk.decrypt(data[0][i][j]))
            A.append(row)
        b = [self.sk.decrypt(ent) for ent in data[1]]
        # solving using numpy solver
        return np.linalg.solve(np.array(A, dtype=np.float), np.array(b, dtype=np.float))


class MLE:
    def __init__(self, qInDO, qInCSP, qOutCSP, n, d, Lambda):
        self.qInDO = qInDO
        self.qInCSP = qInCSP
        self.qOutCSP = qOutCSP
        self.n = n
        self.d = d
        self.Lambda = Lambda

    def Run(self):
        self.phase1()
        self.phase2()

    def phase1(self):
        # get the dataset from all the DOs
        datalst = []
        for qIn in self.qInDO:
            datalst = datalst + qIn.get()
        self.X = []
        self.Y = ([None] * self.d)
        for _ in range(self.d):
            self.X.append([None] * self.n)
        for ent in datalst:
            val, i, j = ent
            if j == 0:
                self.Y[i - 1] = val
            else:
                self.X[i - 1][j - 1] = val

        # get the public key from CSP
        self.pk = self.qInCSP.get()
        self.labhe = labHE(self.pk, F1)
        self.A, self.b = self.qInCSP.get()
        # calculating A,b from encrypted data
        self.A = np.array(self.A)

        self.Aadd = np.dot(np.transpose(self.X), self.X)
        self.A = self.A + self.Aadd

        for i in range(self.n):
            for t in range(self.d):
                self.b[i] = self.b[i] + self.X[t][i] * self.Y[t]

    def phase2(self):
        # sampling random R from GL
        R_arr = []
        R = np.array([[1, 0], [0, 1]])
        while (len(R_arr) == 0) or (np.linalg.det(R) == 0):
            for i in range(self.n):
                row = []
                for _ in range(self.n):
                    row.append(random.uniform(0, 2 ** 20))
                R_arr.append(row)
            R = np.array(R_arr)
        r_arr = []
        for i in range(self.n):
            r_arr.append(random.uniform(0, 2 ** 20))  # int(self.pk.get_random_lt_n()))
        r = np.array(r_arr)
        A = np.array(self.A)
        # masking A by R
        C = np.dot(A, R)
        d = []
        # masking b by r
        for i in range(self.n):
            d.append(self.b[i] + sum([r[k] * A[i][k] for k in range(self.n)]))
        # sending the masked pair to the CSP
        self.qOutCSP.put((C, d))
        # get the result and unmask it
        wTilda = self.qInCSP.get()
        # output stored in self.wStar
        self.wStar = np.dot(R, wTilda) - r


class DO:
    def __init__(self, qInMLE, qOutMLE, qInCSP, qOutCSP):
        self.qInMLE = qInMLE
        self.qOutMLE = qOutMLE
        self.qInCSP = qInCSP
        self.qOutCSP = qOutCSP

    def setData(self, data):
        self.data = data

    def Run(self):
        self.phase1()

    def phase1Hor(self):
        # get the public key from the CSP
        self.pk = self.qInCSP.get()
        X = np.dot(self.data[0], np.transpose(self.data[0]))
        Y = np.dot(self.data[0], self.data[1])
        arr = []
        brr = [None] * len(Y)
        for _ in range(len(X)):
            arr.append([[None] * len(X)])
        for i in range(len(arr)):
            for j in range(len(arr)):
                arr[i][j] = self.pk.encrypt(X[i][j])
        for i in range(len(brr)):
            brr[i] = self.pk.encrypt(Y[i])
        self.qOutMLE.put(arr, brr)

    def phase1(self):
        # get the public key from the CSP
        self.pk = self.qInCSP.get()
        self.labhe = labHE(self.pk, F1)
        # send back sigma encrypted by the public key
        self.qOutCSP.put(self.labhe.pk2)
        msg = []
        # creating the msg to be sent to the MLE
        for ent in self.data:
            val, i, j = ent
            msg.append((self.labhe.labEnc(val, (i, j)), i, j))
        self.qOutMLE.put(msg)


# counting pipe
# this object behaves like one queue apart from the fact that it also count the number of bits send by it and send it to the queue 'sendTo'
class cpipe:
    def __init__(self, sendTo):
        self.sendTo = sendTo
        self.qIn = queue.Queue()
        self.qOut = queue.Queue()
        threading.Thread(target=self.demon, daemon=True).start()

    def demon(self):
        while True:
            msg = self.qIn.get()
            self.sendTo.put(len(pickle.dumps(msg)))
            self.qOut.put(msg)


# getNetWorkSetup create all the pipes and queues needed for communication
# n is the number of Data Owners
def getNetWorkSetup(n):
    # todo
    counting = queue.Queue()
    qMLE_to_CSP = cpipe(counting)
    qCSP_to_MLE = cpipe(counting)
    do_csp_conns = []
    csp_do_conns = []
    do_mle_conns = []
    listForDO = []
    for i in range(n):
        qDO_to_CSP = cpipe(counting)
        qCSP_to_DO = cpipe(counting)
        qDO_to_MLE = cpipe(counting)
        do_csp_conns.append(qDO_to_CSP.qOut)
        csp_do_conns.append(qCSP_to_DO.qIn)
        do_mle_conns.append(qDO_to_MLE.qOut)
        listForDO.append((None, qDO_to_MLE.qIn, qCSP_to_DO.qOut, qDO_to_CSP.qIn))
    connsOfMLE = (do_mle_conns, qMLE_to_CSP.qOut, qCSP_to_MLE.qIn)
    connsOfCSP = (do_csp_conns, csp_do_conns, qCSP_to_MLE.qOut, qMLE_to_CSP.qIn)
    return listForDO, connsOfMLE, connsOfCSP, counting


# network and instances setup
def setUp(m, n, d, Lambda):
    listForDO, connsOfMLE, connsOfCSP, cnt_queue = getNetWorkSetup(m)
    MLE_instance = MLE(connsOfMLE[0], connsOfMLE[1], connsOfMLE[2], n, d, Lambda)
    CSP_instance = CSP(connsOfCSP[0], connsOfCSP[1], connsOfCSP[2], connsOfCSP[3], n, d)
    DOs = []
    for i in range(m):
        connsOfDO_ = listForDO[i]
        DOs.append(DO(connsOfDO_[0], connsOfDO_[1], connsOfDO_[2], connsOfDO_[3]))

    return DOs, MLE_instance, CSP_instance, cnt_queue


# threading and running
def run(DOs, MLE_instance, CSP_instance, cnt_queue):
    t_MLE = threading.Thread(target=MLE_instance.Run)
    t_CSP = threading.Thread(target=CSP_instance.Run)
    t_DOs = []
    for do in DOs:
        t_DOs.append(threading.Thread(target=do.Run))
    for th in (t_DOs + [t_MLE, t_CSP]):
        th.start()
    for th in (t_DOs + [t_MLE, t_CSP]):
        th.join()
    commSum = 0
    while not cnt_queue.empty():
        commSum += cnt_queue.get()
    return DOs, MLE_instance, CSP_instance, commSum


# utility function, splitting the data by the partition and give it to the data owners
def setDataByPartition(DOs, data, partition):
    for idx in range(len(partition)):
        data_i = []
        for ent in partition[idx]:
            if ent[1] == 0:
                data_i.append((data[1][ent[0] - 1], ent[0], ent[1]))
            else:
                data_i.append((data[0][ent[0] - 1][ent[1] - 1], ent[0], ent[1]))
        DOs[idx].setData(data_i)


m = 1  # number of DOs
n = 5  # num of features
d = 5  # num of rows
Lambda = 0
DOs, MLE_instance, CSP_instance, cnt_queue = setUp(m, n, d, Lambda)

features = np.random.rand(d, n)
outputs = np.random.rand(d)
# features = np.array([[1.,0.],[0.,1.],[0.5,0.5]])
# outputs = np.array([1.,4.,2.5])
# creating random partition
partition = []
for i in range(m):
    partition.append([])
for i in range(1, d + 1):
    for j in range(n + 1):
        partition[random.randint(0, m - 1)].append((i, j))

CSP_instance.setPartition(partition)
setDataByPartition(DOs, (features, outputs), partition)
t0 = time.time()
DOs, MLE_instance, CSP_instance, commSum = run(DOs, MLE_instance, CSP_instance, cnt_queue)
print("n=", n, "d=", d)
print("Execution Time:", time.time() - t0, " seconds")
print("Communicaion complexity: ", commSum // 1000, "KB")
print("w*= ",MLE_instance.wStar)
