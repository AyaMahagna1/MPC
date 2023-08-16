import itertools
import random
# triple of Tag, Value, Key (where the key is for the other player's tag,value pair)
import time


class TVK:
    def __init__(self, v, k, k2, p):
        a, b = k
        self.k = k2
        self.p = p
        self.v = v % p
        self.t = (a * v + b) % p

    # overloading of the adding symbol, used as the secret sharing schema is additive
    def __add__(self, b):
        ret = TVK(0, (0, 0), (0, 0), self.p)
        ret.t = (self.t + b.t) % self.p
        ret.v = (self.v + b.v) % self.p
        if self.k[0] != b.k[0]:
            raise Exception('adding with non matching keys!!!')
        ret.k = (self.k[0], self.k[1] + b.k[1])
        return ret

    # used for multiplying the share by a constant
    # the input parameter i is the indicator of which player running this commend
    def mul_const(self, vl, i):
        ret = TVK(0, (0, 0), (0, 0), self.p)
        ret.k = (self.k[0], (self.k[1] * vl) % self.p)
        ret.t = (self.t * vl) % self.p
        ret.v = (self.v * vl) % self.p
        return ret

    def add_const(self, v, i):
        ret = TVK(0, (0, 0), (0, 0), self.p)
        ret.t = self.t
        ret.k = self.k
        ret.v = self.v
        if i == 1:
            ret.k = (self.k[0], (self.k[1] - self.k[0] * v) % self.p)
        else:
            ret.v = (self.v + v) % self.p
        return ret

    # for easy printing of TVK triples
    def __str__(self):
        return "({},{},{})".format(self.v, self.k, self.t)

    # overloading of the & symbol
    # combine two values
    # return null if not valid
    def __and__(self, b):
        if b.t == (self.k[0] * b.v + self.k[1]) % self.p:
            return (self.v + b.v) % self.p
        return None

    # return
    def kless(self):
        ret = TVK(0, (0, 0), (0, 0), self.p)
        ret.t = self.t
        ret.v = self.v
        return ret


def TKVpair(val, alpha1, alpha2, p):
    kA = (alpha1, random.randint(0, p - 1))
    kB = (alpha2, random.randint(0, p - 1))
    valA = random.randint(0, p - 1)
    valB = (val - valA) % p
    r1 = TVK(valA, kA, kB, p)
    r2 = TVK(valB, kB, kA, p)
    return r1, r2


print("Testing TVK for some value of P")
for P in {59, 43, 3, 127}:
    val0, alpha1, alpha2 = random.randint(0, P - 1), random.randint(0, P - 1), random.randint(0, P - 1)
    r1, r2 = TKVpair(val0, alpha1, alpha2, P)
    if r1.t != (r2.k[0] * r1.v + r2.k[1]) % P:
        print("fail1", r1.t, r2.k, r1.v)
    if r2.t != (r1.k[0] * r2.v + r1.k[1]) % P:
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


class BeDOZaDealer:
    def __init__(self, inpSize, size, p):
        self.arrA = []
        self.arrB = []
        self.rOfA = []
        self.rOfB = []
        alpha1, alpha2 = random.randint(0, P - 1), random.randint(0, P - 1)
        for i in range(inpSize):
            a = random.randint(0, p - 1)
            aTag0, aTag1 = TKVpair(a, alpha1, alpha2, p)
            self.rOfA.append(aTag0)
            self.rOfB.append(aTag1)
        for i in range(size):
            a, b = random.randint(0, p - 1), random.randint(0, p - 1)
            c = (a * b) % p
            aTag0, aTag1 = TKVpair(a, alpha1, alpha2, p)
            bTag0, bTag1 = TKVpair(b, alpha1, alpha2, p)
            cTag0, cTag1 = TKVpair(c, alpha1, alpha2, p)
            self.arrA.append((aTag0, bTag0, cTag0))
            self.arrB.append((aTag1, bTag1, cTag1))

    def RandA(self):
        return (self.rOfA, self.arrA)

    def RandB(self):
        return (self.rOfB, self.arrB)


class BeDOZaAgent:
    def __init__(self, i, circ):
        self.circ = circ
        self.i = i

    def Init(self, inp, arrOfR, BeTrip):
        self.orginp = inp
        self.BeTrip = BeTrip
        self.arrOfR = arrOfR
        self.alpha = BeTrip[0][0].k[0]
        self.inp = [None, ] * len(inp)

    def SendInp(self):
        msg = []
        for i in range(len(self.orginp)):
            if self.orginp[i] == None:
                msg.append(self.arrOfR[i].kless())
        return msg

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
        return self.msg2

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

        for node in thisLayer:
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
            if node[0] == "mul":  # we don't support multiplication by constant, this can be done by addition
                u, v, w = self.BeTrip[0]
                self.BeTrip = self.BeTrip[1:]
                d = (x + u)
                e = (y + v)
                self.nextInp.append((d, e, x, y, w))  # save on the node for latter calculations
                msg.append((d.kless(), e.kless()))
        return msg

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
                    raise Exception("d or e are invalid, the communication must end!")
                z = (w + (x.mul_const(e, self.i)) + (y.mul_const(d, self.i)))
                z = z.add_const(-e * d, self.i)
                self.nextInp[i] = z
            i += 1
        self.inp = self.nextInp

    def SendOut(self):
        return self.inp

    def ReceiveOut(self, inp2):
        out = []
        for i in range(len(inp2)):
            out.append((self.inp[i] & inp2[i]))
        return out

    def ended(self):
        return len(self.circ) == 0


# circuit is just layers of arrays of tuples (op,inp1,inp2)
# this circuit test most of the features we support
testCirc = [
    #     0           1            2           3      
    #   x0+y0      x1 + y1      x0 * y0    x1 * y1    
    [("add", 0, 1), ("mul", 2, 3), ("eq", 4, 0), ("add", -1, 5)],
]

# compare >=4 (mod 251)
gt4Circ = [
    [("add", 0, 3 - 251), ("add", 0, 2 - 251), ("add", 0, 1 - 251), ("eq", 0, 0)],
    [("mul", 0, 1), ("mul", 2, 3)],
    [("mul", 0, 1)],  # out
    [("mul", 0, 0)],  # out^2
    [("mul", 0, 0), ("eq", 0, 0)],  # out^4,out^2
    [("mul", 0, 0), ("eq", 1, 1)],  # out^8,out^2
    [("mul", 0, 0), ("mul", 0, 1)],  # out^16,out^10
    [("mul", 0, 0), ("mul", 0, 1)],  # out^32,out^26
    [("mul", 0, 0), ("mul", 0, 1)],  # out^64,out^58
    [("mul", 0, 0), ("mul", 0, 1)],  # out^128,out^122
    [("mul", 0, 1)]  # out^250
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


def appendHorizontally(appTo, circ, lvl):
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
    [("mul", 0, 2), ("mul", 1, 3)],
    [("add", 0, 1)],
]
Equation3 = Equation3 + gt4Circ


def mainTest(x, y, circ, p=59):
    communication = 0
    Dealer = BeDOZaDealer(len(x), mulSizeOfCirc(circ), p)
    Alice, Bob = BeDOZaAgent(1, circ), BeDOZaAgent(2, circ)
    Alice.Init(x, Dealer.RandA()[0], Dealer.RandA()[1])
    Bob.Init(y, Dealer.RandB()[0], Dealer.RandB()[1])
    msg1 = Alice.SendInp()
    msg2 = Bob.SendInp()
    communication += len(msg1) + len(msg2)
    Bob.ReceiveInp(msg1)
    Alice.ReceiveInp(msg2)
    msg1 = Alice.SendInp2()
    msg2 = Bob.SendInp2()
    communication += len(msg1) + len(msg2)
    Bob.ReceiveInp2(msg1)
    Alice.ReceiveInp2(msg2)
    #     print("inp:",Alice.inp[0],Bob.inp[0])
    while (not Alice.ended()):
        #     print("layer")
        msg1 = Alice.Send()
        msg2 = Bob.Send()
        communication += len(msg1) + len(msg2)
        Bob.Receive(msg1)
        Alice.Receive(msg2)
    msg1 = Bob.SendOut()
    communication += len(msg1)
    z = Alice.ReceiveOut(msg1)
    return z, communication


p = 11
print("Testing all the basic operations with test circuit")
for x1, x2, x3, y1, y2, y3 in itertools.product(range(0, 6), repeat=6):
    x = [x1, None, x2, None, x3, None]
    y = [None, y1, None, y2, None, y3]
    out, _ = mainTest(x, y, testCirc, p)
    if (out[0]) != (x1 + y1) % p:
        print("+", x1, y1, out[0])
    if (out[1]) != (x2 * y2) % p:
        print("*", x2, y2, out[1])
    if (out[2]) != (x3) % p:
        print("=", x3, out[2])
    if (out[3]) != (y3 + 1) % p:
        print("++", y3, out[3])

commCnt = 0
maxComm = 0
i = 0
p = 251
start_time = time.time()
print("Testing the Equation3 (for 100 times on every input, may take a minute)")
for trys in range(100):
    print(trys, end='\r')
    for x1, x2, y1, y2 in itertools.product(range(0, 4), repeat=4):
        x = [x1, x2, None, None]
        y = [None, None, y1, y2]
        out, cc = mainTest(x, y, Equation3, p)
        commCnt += cc
        i += 1
        if maxComm < cc:
            maxComm = cc
        #         print(out)
        if not (out[0] in {0, 1}):
            print(x1, x2, y1, y2)
        if (out[0] == 1) != (x1 * y1 + x2 * y2 >= 4):
            print(x1, x2, y1, y2)
print("The average and maximal communication per activation", commCnt / i, maxComm)
print("---Execution time for all the 100 possible inputs is: %s seconds ---" % (time.time() - start_time))
