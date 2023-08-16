import itertools
import pickle
import queue
import random
import threading
import time
from random import randrange


def is_prime(n, k=10):
    if n == 2:
        return True
    if not n & 1:
        return False

    def check(a, s, d, n):
        x = pow(a, d, n)
        if x == 1:
            return True
        for i in range(s - 1):
            if x == n - 1:
                return True
            x = pow(x, 2, n)
        return x == n - 1

    s = 0
    d = n - 1

    while d % 2 == 0:
        d >>= 1
        s += 1

    for i in range(k):
        a = randrange(2, n - 1)
        if not check(a, s, d, n):
            return False
    return True


# adapted such that the prime is of the form p=2*q+1
def next_nice_prime(num, k=40):
    if (not num & 1) and (num != 2):
        num += 1
    while True:
        if is_prime(num, k) and is_prime(2 * num + 1, k):
            break
        num += 2
    return num


class PKE_ElGamal:
    def __init__(this, n):
        this.p = next_nice_prime(random.randint(1 << n, 1 << (n + 1)), n // 2)
        this.q = 2 * this.p + 1

    def Gen(this):
        alpha = random.randint(0, this.p - 1)
        pk = (4, pow(4, alpha, this.q))
        sk = alpha
        return (pk, sk)

    def OGen(this):
        s = random.randint(0, this.p - 1)
        pk = (4, pow(s, 2, this.q))
        return (pk, None)

    def enc(this, pk, m):
        g, h = pk
        r = random.randint(0, this.p - 1)
        # we need to convert m into qudrtic element of G, as long as m <= p this is one-to-one
        return (pow(g, r, this.q), (pow(m, 2) * pow(h, r, this.q)) % this.q)

    def dec(this, sk, C):
        c1, c2 = C
        ret = pow((c2 * pow(c1, (this.p - sk), this.q)), (this.q + 1) // 4, this.q)
        if ret > this.p:
            return this.q - ret
        return ret  # x^(p-sk) is equivelent to 1/(x^sk)


print("Testing PKE-ElGamal")
PKE1 = PKE_ElGamal(100)
(pk, sk) = PKE1.Gen()
C = PKE1.enc(pk, 12032)
if PKE1.dec(sk, C) != 12032:
    print("fail")


class OT:
    def __init__(this, PKE):
        this.PKE = PKE

    def load(this, msg1, msg2):
        this.data = [msg1, msg2]

    def getKeys(self, keys):
        pk1, pk2 = keys
        self.msg = [self.PKE.enc(pk1, self.data[0]), self.PKE.enc(pk2, self.data[1])]

    def sendMsgs(self):
        return self.msg

    def chose(self, j):
        ret = [None, None]
        pk1, sk = self.PKE.Gen()
        pk2, _ = self.PKE.OGen()
        ret[j] = pk1
        ret[1 - j] = pk2
        self.sk = sk
        self.j = j
        self.ret = ret

    def sendChose(self):
        return self.ret

    def getData(self, msg):
        return self.PKE.dec(self.sk, msg[self.j])

    def runOn1(self, qI, qO):
        self.getKeys(qI.get())
        qO.put(self.sendMsgs())

    def runOn2(self, qI, qO):
        qO.put(self.sendChose())
        return self.getData(qI.get())


print("Testing 1-out-of-2 OT (with PKE of ElGamal)")
PKE2 = PKE_ElGamal(20)
ot1 = OT(PKE2)
ot2 = OT(PKE2)
ot1.load(10, 20)
c = random.randint(0, 1)
ot2.chose(c)
q1, q2 = queue.Queue(), queue.Queue()


def f2():
    if (ot2.runOn2(q1, q2)) != c * 10 + 10:
        print("fail")


def f1():
    ot1.runOn1(q2, q1)


w = threading.Thread(name='ot1', target=f1)
w2 = threading.Thread(name='ot2', target=f2)
w.start()
w2.start()
w.join()
w2.join()


# triple of Tag, Value, Key (where the key is for the other player's tag,value pair)
# this can be extended to multiply tags and keys.
class TVK:
    def __init__(self, v, k, k2, p):
        self.k = k2
        self.p = p
        self.v = v % p
        self.t = []
        for a, b in k:
            self.t.append((a * v + b) % p)

    # overloding of the adding symbole, used as the secret sharring scheam is addative
    def __add__(self, b):
        ret = TVK(0, [(0, 0)], [(0, 0)], self.p)
        ret.t = [(self.t[i] + b.t[i]) % self.p for i in range(len(self.t))]
        ret.v = (self.v + b.v) % self.p
        for i in range(len(self.t)):
            if self.k[i][0] != b.k[i][0]:
                raise Exception('adding with non matching keys!!!')
        ret.k = [(self.k[i][0], (self.k[i][1] + b.k[i][1]) % self.p) for i in range(len(self.t))]
        return ret

    # used for multiplying the shere by a constant
    # the input paramter i is the indectior of which player running this commend
    def mul_const(self, vl, i):
        ret = TVK(0, [(0, 0)], [(0, 0)], self.p)
        #         ret.t = self.t
        #         ret.k = self.k
        #         ret.v = self.v
        #         if i == 1:
        ret.k = [(self.k[i][0], (self.k[i][1] * vl) % self.p) for i in range(len(self.t))]
        #         else:
        ret.t = [(t * vl) % self.p for t in self.t]
        ret.v = (self.v * vl) % self.p
        return ret

    def add_const(self, v, i):
        ret = TVK(0, [(0, 0)], [(0, 0)], self.p)
        ret.t = self.t
        ret.k = self.k
        ret.v = self.v
        if i == 1:
            ret.k = [(self.k[i][0], (self.k[i][1] - self.k[i][0] * v) % self.p) for i in range(len(self.t))]
        else:
            ret.v = (self.v + v) % self.p
        return ret

    # for easy printing of TVK triples
    def __str__(self):
        return "({},{},{})".format(self.v, self.k, self.t)

    # overloding of the & symbole
    # combain two values
    # return null if not valid
    def __and__(self, b):
        if all([b.t[i] == (self.k[i][0] * b.v + self.k[i][1]) % self.p for i in range(len(self.t))]):
            return (self.v + b.v) % self.p
        return None

    # return
    def kless(self):
        ret = TVK(0, [(0, 0)], [(0, 0)], self.p)
        ret.t = self.t
        ret.v = self.v
        return ret


def TKVpair(val, alpha1, alpha2, p):
    kA = list(zip(alpha1, [random.randint(0, p - 1) for _ in range(len(alpha1))]))
    kB = list(zip(alpha2, [random.randint(0, p - 1) for _ in range(len(alpha2))]))
    valA = random.randint(0, p - 1)
    valB = (val - valA) % p
    r1 = TVK(valA, kA, kB, p)
    r2 = TVK(valB, kB, kA, p)
    return r1, r2


def withAuth(alpha, x, k2, p=2):
    k = zip(alpha, k2)
    return TVK(x, k, None, p)


print("Testing TVK for some value of P")
for P in {2, 2, 2, 2, 2, 11, 127, 53}:
    val0, alpha1, alpha2 = random.randint(0, P - 1), [random.randint(0, P - 1)], [random.randint(0, P - 1)]
    r1, r2 = TKVpair(val0, alpha1, alpha2, P)
    if r1.t[0] != (r2.k[0][0] * r1.v + r2.k[0][1]) % P:
        print("fail1", r1.t, r2.k, r1.v)
    if r2.t[0] != (r1.k[0][0] * r2.v + r1.k[0][1]) % P:
        print("fail2")
    if (r1.v + r2.v) % P != val0:
        print("fail3")
    val1 = random.randint(0, P - 1)
    r3, r4 = TKVpair(val1, alpha1, alpha2, P)
    r5, r6 = r1 + r3, r2 + r4
    r7, r8 = r1.add_const(3, 1), r2.add_const(3, 2)
    r9, r10 = r1.mul_const(10, 1), r2.mul_const(10, 2)
    if r1 & r2 != val0:
        print("ERROR1")
    if r5 & r6 != (val0 + val1) % P:
        print("ERROR2", r5 & r6, (val0 + val1) % P)
    if r7 & r8 != (val0 + 3) % P:
        print("ERROR3")
    if r9 & r10 != (val0 * 10) % P:
        print("ERROR4", r1, r2, r9, r10, r9 & r10)


# util functions for PKE and OT
# convert some structures into numbers and backward
def encodeArray(arr, p):
    return sum([arr[i] * (p ** i) for i in range(len(arr))])


def decodeArray(v, p, l):
    return [(v // (p ** i)) % p for i in range(l)]


def encodeShare(share, p):
    return encodeArray([share.v] + share.t, p)


def decodeShare(v, l, p):
    arr = decodeArray(v, p, l)
    tkv = TVK(arr[0], [(0, 0)], None, p)
    tkv.t = arr[1:]
    return tkv


def decodeShare2(arr, p):
    tkv = TVK(arr[0], [(0, 0)], None, p)
    tkv.t = arr[1:]
    return tkv


class BeDOZaAgent:
    def __init__(self, i, circ, PKE, qi, qo, sc_pram=20):
        self.circ = circ
        self.i = i
        self.PKE = PKE
        self.qOut = qo
        self.qIn = qi
        self.secPram = sc_pram
        self.alpha = [random.randint(0, 1) for _ in range(self.secPram)]

    def initOTsOfR(self, length):
        self.listOfOT1_s = []
        self.listOfOT1_r = []
        self.arrOfR = []
        for i in range(length):
            self.listOfOT1_s.append(OT(self.PKE))
            k = [random.randint(0, 1) for i in range(self.secPram)]
            self.listOfOT1_s[-1].load(encodeShare(withAuth(self.alpha, 0, k), 2),
                                      encodeShare(withAuth(self.alpha, 1, k), 2))
            self.listOfOT1_r.append(OT(self.PKE))
            self.listOfOT1_r[-1].chose(random.randint(0, 1))
            self.arrOfR.append(k)

    def runOTsOfR(self):
        length = len(self.listOfOT1_s)
        for i in range(length):
            if self.i == 0:
                self.listOfOT1_s[i].runOn1(self.qIn, self.qOut)
                out = decodeShare(self.listOfOT1_r[i].runOn2(self.qIn, self.qOut), self.secPram + 1, 2)
            else:
                out = decodeShare(self.listOfOT1_r[i].runOn2(self.qIn, self.qOut), self.secPram + 1, 2)
                self.listOfOT1_s[i].runOn1(self.qIn, self.qOut)
            k = self.arrOfR[i]
            out.k = list(zip(self.alpha, k))
            self.arrOfR[i] = out

    # done only by alice (i=0)
    def initOTsOfBeTrip(self, length):
        self.listOfOT2 = []
        self.listOfOT3 = []
        tmp = self.arrOfR[-length * 3:]
        self.arrOfR = self.arrOfR[:-length * 3]
        self.BeTrip = [(tmp[3 * i], tmp[3 * i + 1], tmp[3 * i + 2]) for i in range(length)]
        if self.i == 0:
            for i in range(length):
                self.listOfOT2.append(OT(self.PKE))
                self.listOfOT3.append(OT(self.PKE))
                u, v, w = self.BeTrip[i]
                snd = []
                msk = [[random.randint(0, 1) for _ in range(self.secPram + 1)],
                       [random.randint(0, 1) for _ in range(self.secPram + 1)]]
                for u_, v_ in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    sendW = ((u.v + u_) * (v.v + v_) + w.v) % 2
                    share = withAuth(self.alpha, sendW, [b for _, b in w.k])
                    msg = [share.v] + share.t
                    snd = snd + [(msg[i] + msk[v_][i]) % 2 for i in range(self.secPram + 1)]
                self.listOfOT2[-1].load(encodeArray(snd[:2 * self.secPram + 2], 2),
                                        encodeArray(snd[2 * self.secPram + 2:], 2))
                self.listOfOT3[-1].load(encodeArray(msk[0], 2), encodeArray(msk[1], 2))
        else:
            for i in range(length):
                self.listOfOT2.append(OT(self.PKE))
                self.listOfOT3.append(OT(self.PKE))
                u, v, _ = self.BeTrip[i]
                self.listOfOT2[-1].chose(u.v)
                self.listOfOT3[-1].chose(v.v)

    def runOTsOfBeTrip(self):
        length = len(self.BeTrip)
        if self.i == 0:
            for i in range(length):
                self.listOfOT2[i].runOn1(self.qIn, self.qOut)
                self.listOfOT3[i].runOn1(self.qIn, self.qOut)
        if self.i == 1:
            for i in range(length):
                u, v, w = self.BeTrip[i]
                pair = self.listOfOT2[i].runOn2(self.qIn, self.qOut)
                arr = decodeArray(pair, 2, 2 * (self.secPram + 1))
                arr2 = arr[v.v * (self.secPram + 1):(v.v + 1) * (self.secPram + 1)]
                msk = decodeArray(self.listOfOT3[i].runOn2(self.qIn, self.qOut), 2, (self.secPram + 1))
                tvk = decodeShare2([(msk[i] + arr2[i]) % 2 for i in range(self.secPram + 1)], 2)
                tvk.k = w.k
                self.BeTrip[i] = (u, v, tvk)

    def Init(self, lr, lb):
        self.initOTsOfR(lr + lb * 3)
        self.runOTsOfR()
        self.initOTsOfBeTrip(lb)
        self.runOTsOfBeTrip()

    def enterInp(self, inp):
        self.orginp = inp
        self.inp = [None, ] * len(inp)

    def Run(self):
        qIn = self.qIn
        if self.i == 0:
            self.SendInp()
            self.ReceiveInp(qIn.get())
        else:
            self.ReceiveInp(qIn.get())
            self.SendInp()

        if self.i == 0:
            self.SendInp2()
            self.ReceiveInp2(qIn.get())
        else:
            self.ReceiveInp2(qIn.get())
            self.SendInp2()
        while (not self.ended()):
            self.Send()
            self.Receive(qIn.get())

        if self.i == 0:
            self.SendOut()
        else:
            self.out = self.ReceiveOut(qIn.get())

    def SendInp(self):
        msg = []
        for i in range(len(self.orginp)):
            if self.orginp[i] == None:
                msg.append(self.arrOfR[i].kless())
        #         return msg
        self.qOut.put(msg)

    def ReceiveInp(self, inp2):
        self.msg2 = []
        for i in range(len(self.orginp)):
            if self.orginp[i] != None:
                r = self.arrOfR[i] & inp2[0]
                inp2 = inp2[1:]
                d = self.orginp[i] - r
                self.msg2.append(d)
                self.inp[i] = self.arrOfR[i].add_const(d, self.i)

    def SendInp2(self):
        self.qOut.put(self.msg2)

    def ReceiveInp2(self, inp2):
        for i in range(len(self.orginp)):
            if self.orginp[i] == None:
                self.inp[i] = self.arrOfR[i].add_const(inp2[0], self.i)
                inp2 = inp2[1:]

    def Send(self):
        thisLayer = self.circ[0]
        self.circ = self.circ[1:]
        self.nextInp = []
        msg = []

        #         print(thisLayer)
        for node in thisLayer:
            # calc the node
            if node[1] >= 0:
                x = self.inp[node[1]]
            if node[2] >= 0:
                y = self.inp[node[2]]
            if node[0] == "add":
                if node[2] < 0:
                    self.nextInp.append(x.add_const(-node[2], self.i))
                elif node[1] < 0:
                    self.nextInp.append(y.add_const(-node[1], self.i))
                else:
                    self.nextInp.append(x + y)
            if node[0] == "eq":
                self.nextInp.append(x)
            if node[0] == "mul":
                # mul case
                u, v, w = self.BeTrip[0]
                self.BeTrip = self.BeTrip[1:]
                d = (x + u)
                e = (y + v)
                self.nextInp.append((d, e, x, y, w))  # save on the node for latter calculations
                msg.append((d.kless(), e.kless()))
        self.qOut.put(msg)

    def Receive(self, msg):
        i = 0
        for node in self.nextInp:
            if type(node) == type((0,)):
                dTag0, eTag0 = msg[0]
                msg = msg[1:]
                dTag1, eTag1, x, y, w = node
                d, e = (dTag1 & dTag0), (eTag1 & eTag0)
                if d is None or e is None:
                    print(d, dTag1.v, dTag1.k, dTag1.t, dTag0.v, dTag0.k, dTag0.t)
                    raise Exception("d or e are invalide, the communiction must end!")
                z = (w + (x.mul_const(e, self.i)) + (y.mul_const(d, self.i)))
                z = z.add_const(-e * d, self.i)
                self.nextInp[i] = z
            i += 1
        self.inp = self.nextInp

    def SendOut(self):
        self.qOut.put(self.inp)

    def ReceiveOut(self, inp2):
        out = []
        for i in range(len(inp2)):
            out.append((self.inp[i] & inp2[i]))
        return out

    def ended(self):
        return len(self.circ) == 0


# circuit is just layers of arrays of tuppels (op,inp1,inp2)
# do some testing
testCirc = [
    #     0           1            2           3      
    #   x0+x1      x2 * x3        x4        x5 + 1    
    [("add", 0, 1), ("mul", 2, 3), ("eq", 4, 0), ("add", -1, 5)],
]

# add two numbers in the range (0-7)
sumCirc = [
    #     0           1            2           3          4          5          6           7
    #   x0+y0      x1 + y1      x0 * y0    x1 * y1       x1        x2 + 1       y1        1 + y2
    [("add", 0, 3), ("add", 1, 4), ("mul", 0, 3), ("mul", 1, 4), ("eq", 1, 1), ("add", 2, -1), ("eq", 4, 4),
     ("add", 5, -1)],
    #   Out1        Out2      1 + x1*y1    x1*x0*y0    y1*x0*y0    (x2 + 1) * (1 + y2)
    [("eq", 0, 0), ("add", 1, 2), ("add", 3, -1), ("mul", 4, 2), ("mul", 6, 2), ("mul", 5, 7)],
    #   Out1        Out2   1 + x1*x0*y0|1 + y1*x0*y0|   ???
    [("eq", 0, 0), ("eq", 1, 1), ("add", 3, -1), ("add", 4, -1), ("mul", 2, 5)],
    [("eq", 0, 0), ("eq", 1, 1), ("mul", 2, 3), ("eq", 4, 4)],
    [("eq", 0, 0), ("eq", 1, 1), ("mul", 2, 3)],
    [("eq", 0, 0), ("eq", 1, 1), ("add", 2, -1)]
]


def changeVars(dic, lyr):
    update = lambda x: dic[x] if x >= 0 else -1
    for i in range(len(lyr)):
        lyr[i] = (lyr[i][0], update(lyr[i][1]), update(lyr[i][2]))
    return lyr


def addToVars(k, lyr):
    update = lambda x: x + k if x >= 0 else -1
    for i in range(len(lyr)):
        lyr[i] = (lyr[i][0], update(lyr[i][1]), update(lyr[i][2]))
    return lyr


def appendHorizontaly(appTo, circ, lvl):
    for i in range(lvl, lvl + len(circ)):
        if len(appTo) <= i:
            appTo = appTo + [[]]  # adding empty layer
        lyr2 = circ[i - lvl]
        if i > lvl:
            lyr2 = addToVars(len(appTo[i - 1]) - len(circ[i - lvl - 1]), lyr2)
        appTo[i] = appTo[i] + (lyr2)
    return appTo


def copyOf(circ):
    circ2 = []
    for lyr in circ:
        l = []
        for node in lyr:
            l.append((node[0], node[1], node[2]))
        circ2.append(l)
    return circ2


def mulSizeOfCirc(circ):
    cnt = 0
    for lyr in circ:
        for node in lyr:
            if node[0] == "mul":
                cnt += 1
    return cnt


Equation3 = [
    #         0             1          2            3           4          5            6           7            8
    [("add", 1, 1), ("mul", 0, 4), ("mul", 0, 5), ("mul", 1, 4), ("mul", 1, 5), ("mul", 2, 6), ("mul", 2, 7),
     ("mul", 3, 6), ("mul", 3, 7)]
]

s1 = copyOf(sumCirc)
s2 = copyOf(sumCirc)
s3 = copyOf(sumCirc)
s1[0] = changeVars({0: 1, 1: 2, 2: 0, 3: 0, 4: 3, 5: 4}, s1[0])
s2[0] = changeVars({0: 5, 1: 6, 2: 0, 3: 0, 4: 7, 5: 8}, s2[0])
s3[0] = changeVars({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}, s3[0])
Equation3 = appendHorizontaly(Equation3, s1, 1)
Equation3 = appendHorizontaly(Equation3, s2, 1)
Equation3 = appendHorizontaly(Equation3, s3, 7)
Equation3.append([("eq", 2, 2)])
Equation3

print("Testing BeDOZa on simple circuit")
lr, lb = 6, 1
sc_prm = 10
for x0, x1, x2, x3, x4, x5 in itertools.product(range(2), repeat=6):
    PKE1 = PKE_ElGamal(sc_prm * 3)
    q1, q2 = queue.Queue(maxsize=2), queue.Queue(maxsize=2)
    a = BeDOZaAgent(0, testCirc, PKE1, q1, q2, sc_prm)
    b = BeDOZaAgent(1, testCirc, PKE1, q2, q1, sc_prm)
    t1 = threading.Thread(target=a.Init, args=(lr, lb))
    t2 = threading.Thread(target=b.Init, args=(lr, lb))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    if not all([a.arrOfR[i] & b.arrOfR[i] != None for i in range(lr)]):
        print("fail")
    for i in range(lb):
        v = (a.BeTrip[i][0] & b.BeTrip[i][0])
        u = (a.BeTrip[i][1] & b.BeTrip[i][1])
        w = (a.BeTrip[i][2] & b.BeTrip[i][2])
        if 0 != (v * u - w):
            print("fail")
    a.enterInp([None, x1, None, x3, None, x5])
    b.enterInp([x0, None, x2, None, x4, None])
    t1 = threading.Thread(target=a.Run)
    t2 = threading.Thread(target=b.Run)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    if b.out[0] != (x0 + x1) % 2:
        print("fail")
    if b.out[1] != x2 * x3:
        print("fail")
    if b.out[2] != x4:
        print("fail")
    if b.out[3] != (x5 + 1) % 2:
        print("fail")

print("Testing BeDOZa on Equation3 (may take a minute)")
lr, lb = 8, mulSizeOfCirc(Equation3)
sec_pram = 6
for x0, x1, x2, x3, x4, x5, x6, x7 in itertools.product(range(2), repeat=8):
    # if x4 == x5 == x6 == x7 == 0:
    #     print(x0,x1,x2,x3)
    PKE1 = PKE_ElGamal(sec_pram * 3)
    q1, q2 = queue.Queue(maxsize=2), queue.Queue(maxsize=2)
    a = BeDOZaAgent(0, Equation3, PKE1, q1, q2, sec_pram)
    b = BeDOZaAgent(1, Equation3, PKE1, q2, q1, sec_pram)
    t1 = threading.Thread(target=a.Init, args=(lr, lb))
    t2 = threading.Thread(target=b.Init, args=(lr, lb))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    #     print("end step 1")
    if not all([a.arrOfR[i] & b.arrOfR[i] != None for i in range(lr)]):
        print("fail")
    for i in range(lb):
        v = (a.BeTrip[i][0] & b.BeTrip[i][0])
        u = (a.BeTrip[i][1] & b.BeTrip[i][1])
        w = (a.BeTrip[i][2] & b.BeTrip[i][2])
        if 0 != (v * u - w):
            print("fail")
    # start of "online" phase
    a.enterInp([x0, x1, x2, x3, None, None, None, None])
    b.enterInp([None, None, None, None, x4, x5, x6, x7])
    t1 = threading.Thread(target=a.Run)
    t2 = threading.Thread(target=b.Run)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    if (b.out[0] == 1) != ((x0 + x1 * 2) * (x4 + x5 * 2) + (x2 + x3 * 2) * (x6 + x7 * 2) >= 4):
        print(x0, x1, x2, x3, x4, x5, x6, x7)


def moveAndSum(qIn, qOut, qSum):
    while True:
        c = qIn.get()
        qSum.put(len(pickle.dumps(c)))
        qOut.put(c)

start_time = time.time()
print("profiling BeDOZa communication complexity, in respect to the security parameter (on Equation3)")
print("prm|comm(kB)")
for sec_pram in range(10, 45, 5):
    avg = 0
    for i in range(5):
        x0, x1, x2, x3, x4, x5, x6, x7 = random.randint(0, 1), random.randint(0, 1), random.randint(0,
                                                                                                    1), random.randint(
            0, 1), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)
        PKE1 = PKE_ElGamal(sec_pram * 3)
        q1, q2 = queue.Queue(maxsize=2), queue.Queue(maxsize=2)
        q3, q4 = queue.Queue(maxsize=2), queue.Queue(maxsize=2)
        sumQ = queue.Queue(maxsize=-1)
        a = BeDOZaAgent(0, Equation3, PKE1, q1, q2, sec_pram)
        b = BeDOZaAgent(1, Equation3, PKE1, q4, q3, sec_pram)
        t1 = threading.Thread(target=a.Init, args=(lr, lb))
        t2 = threading.Thread(target=b.Init, args=(lr, lb))
        t3 = threading.Thread(target=moveAndSum, args=(q3, q1, sumQ), daemon=True)
        t4 = threading.Thread(target=moveAndSum, args=(q2, q4, sumQ), daemon=True)
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t1.join()
        t2.join()
        commSum = 0
        while not sumQ.empty():
            commSum += sumQ.get()
        avg += commSum
        a.enterInp([x0, x1, x2, x3, None, None, None, None])
        b.enterInp([None, None, None, None, x4, x5, x6, x7])
        t1 = threading.Thread(target=a.Run)
        t2 = threading.Thread(target=b.Run)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        if (b.out[0] == 1) != ((x0 + x1 * 2) * (x4 + x5 * 2) + (x2 + x3 * 2) * (x6 + x7 * 2) >= 4):
            print(x0, x1, x2, x3, x4, x5, x6, x7)
    print(sec_pram, "|", ((avg / 5) // 1000) / 10, "kB")
print("---Execution time is: %s seconds ---" % (time.time() - start_time))
