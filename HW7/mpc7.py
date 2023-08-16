import hashlib
import itertools
import os
import pickle
import queue
import random
import threading
import time

from ipython_genutils.py3compat import xrange


def modular_sqrt(a, p):
    """ Find a quadratic residue (mod p) of 'a'. p
        must be an odd prime.

        Solve the congruence of the form:
            x^2 = a (mod p)
        And returns x. Note that p - x is also a root.

        0 is returned is no square root exists for
        these a and p.

        The Tonelli-Shanks algorithm is used (except
        for some simple cases in which the solution
        is known from an identity). This algorithm
        runs in polynomial time (unless the
        generalized Riemann hypothesis is false).
    """
    # Simple cases
    #
    if legendre_symbol(a, p) != 1:
        return 0
    elif a == 0:
        return 0
    elif p == 2:
        return 0
    elif p % 4 == 3:
        return

    # Partition p-1 to s * 2^e for an odd s (i.e.
    # reduce all the powers of 2 from p-1)
    #
    s = p - 1
    e = 0
    while s % 2 == 0:
        s /= 2
        e += 1

    # Find some 'n' with a legendre symbol n|p = -1.
    # Shouldn't take long.
    #
    n = 2
    while legendre_symbol(n, p) != -1:
        n += 1

    # Here be dragons!
    # Read the paper "Square roots from 1; 24, 51,
    # 10 to Dan Shanks" by Ezra Brown for more
    # information
    #

    # x is a guess of the square root that gets better
    # with each iteration.
    # b is the "fudge factor" - by how much we're off
    # with the guess. The invariant x^2 = ab (mod p)
    # is maintained throughout the loop.
    # g is used for successive powers of n to update
    # both a and b
    # r is the exponent - decreases with each update
    #
    x = pow(a, (s + 1) / 2, p)
    b = pow(a, s, p)
    g = pow(n, s, p)
    r = e

    while True:
        t = b
        m = 0
        for m in xrange(r):
            if t == 1:
                break
            t = pow(t, 2, p)

        if m == 0:
            return x

        gs = pow(g, 2 ** (r - m - 1), p)
        g = (gs * gs) % p
        x = (x * gs) % p
        b = (b * g) % p
        r = m


def legendre_symbol(a, p):
    """ Compute the Legendre symbol a|p using
        Euler's criterion. p is a prime, a is
        relatively prime to p (if p divides
        a, then a|p = 0)

        Returns 1 if a has a square root modulo
        p, -1 otherwise.
    """
    ls = pow(a, (p - 1) / 2, p)
    return -1 if ls == p - 1 else ls


# due to https://stackoverflow.com/questions/17298130/working-with-large-primes-in-python
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


# addpeted such that the prime is of the form p=2*q+1
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


# we use a soultion to xor of bytes strings from https://stackoverflow.com/questions/2612720/how-to-do-bitwise-exclusive-or-of-two-strings-in-python/28481974#28481974
def bytes_xor(a, b):
    return bytes(x ^ y for x, y in zip(a, b))


# do random permutation of array a
def Perm(a):
    random.shuffle(a)
    return a


# encode label x with two others labels k1,k2
def Encode(x, k1, k2):
    r = label(x.l1, len(x.val))
    r.val = bytes_xor(x.val, cryptohash(k1, k2))
    return r


def Decode(y, k1, k2):
    r = label(y.l1, len(y.val))
    r.val = bytes_xor(y.val, cryptohash(k1, k2))
    return r


def cryptohash(k1, k2):
    if k2 == 0:
        return hashlib.sha256(k1.val).digest()
    return hashlib.sha256(k1.val + k2.val).digest()


# label is value (bytes)
class label():
    def __init__(self, l1, l2):
        self.val = os.urandom(l1) + bytes(l2 - l1)
        self.l1 = l1

    # valid chacks that all the bytes after the first l1 bytes are zero
    def valid(self):
        return all([self.val[i] == 0 for i in range(self.l1, len(self.val))])


# labelTuple is two labels that represent the True,False values of some wire
class labelTuple():
    def __init__(self, l1, l2):
        self.v0 = label(l1, l2)
        self.v1 = label(l1, l2)

    def get(self, i):
        if i == 0:
            return self.v0
        return self.v1


# genTT creates the gate of Garbled Circut from two inputs wire inp1,inp2
# genTT support 'or','xor','add','and','mul' and 'not'
# all of the above are mod 2
def genTT(node_type, inp1, inp2, l1, l2):
    out = labelTuple(l1, l2)
    if node_type == "or":
        return Perm([
            Encode(out.v0, inp1.v0, inp2.v0),
            Encode(out.v1, inp1.v0, inp2.v1),
            Encode(out.v1, inp1.v1, inp2.v0),
            Encode(out.v1, inp1.v1, inp2.v1)
        ]), out
    if node_type == "add" or node_type == "xor":
        return Perm([
            Encode(out.v0, inp1.v0, inp2.v0),
            Encode(out.v1, inp1.v0, inp2.v1),
            Encode(out.v1, inp1.v1, inp2.v0),
            Encode(out.v0, inp1.v1, inp2.v1)
        ]), out
    elif node_type == "and" or node_type == "mul":
        return Perm([
            Encode(out.v0, inp1.v0, inp2.v0),
            Encode(out.v0, inp1.v0, inp2.v1),
            Encode(out.v0, inp1.v1, inp2.v0),
            Encode(out.v1, inp1.v1, inp2.v1)
        ]), out
    # node_type == "not"
    return Perm([
        Encode(out.v1, inp1.v0, 0),
        Encode(out.v0, inp1.v1, 0)
    ]), out


# Garbled Circut
class GC:
    # as hashlib.sha256 work with 32 bytes (512 = 32*8 bits) we choose as default l2=32
    def __init__(this, circ, inpSize, l1, l2=32):
        this.circ = []
        this.wire = []
        w = []
        for n in range(inpSize):
            w.append(labelTuple(l1, l2))
        this.wire.append(w)
        for lyr in circ:
            l = []
            w = []
            for nd in lyr:
                (op, x, y) = nd
                gate, out = genTT(op, this.wire[-1][x], this.wire[-1][y], l1, l2)
                w.append(out)
                l.append((gate, x, y))
            this.circ.append(l)
            this.wire.append(w)

    def DecodeOutput(this, lbl):
        if this.wire[-1][0].get(0).val == lbl.val:
            return 0
        return 1

    def propgate(this, inp, lvl):
        ret = []
        for nd in this.circ[lvl]:
            gate, x, y = nd
            ret.append(evalGate(gate, inp[x], inp[y]))
        return ret

    def preaperForSend(self):
        self.wire = [self.wire[-1]]


def evalGate(gate, inpx, inpy):
    for r in gate:
        l = Decode(r, inpx, inpy)
        if l.valid():
            return l
    for r in gate:
        l = Decode(r, inpx, 0)
        if l.valid():
            return l
    return None


class Alice():
    def __init__(self, qIn, qOut, pke):
        self.qOut = qOut
        self.qIn = qIn
        self.PKE = pke

    def setInps(self, inpArr):
        self.inp = inpArr

    def setCirc(self, circ, inpSize, l1):
        self.gc = GC(circ, inpSize, l1)
        self.l1 = l1

    def giveInps(self):
        self.qOut.put(self.l1)
        for i in range(len(self.inp)):
            if self.inp[i] == None:
                ot = OT(self.PKE)
                ot.load(int.from_bytes(self.wire0[i].get(0).val, "little"),
                        int.from_bytes(self.wire0[i].get(1).val, "little"))
                ot.runOn1(self.qIn, self.qOut)

    def sendCirc(self):
        gc = self.gc
        self.wire0 = self.gc.wire[0]
        gc.preaperForSend()
        self.qOut.put(gc)

    def sendInp(self):
        snd = []
        for i in range(len(self.inp)):
            if self.inp[i] != None:
                snd.append(self.gc.wire[0][i].get(self.inp[i]))
            else:
                snd.append(None)
        self.qOut.put(snd)

    def Run(self):
        self.sendInp()
        self.sendCirc()
        self.giveInps()


class Bob():
    def __init__(self, qIn, qOut, pke):
        self._inp = []
        self.qIn = qIn
        self.qOut = qOut
        self.PKE = pke

    def setCirc(self):
        self.gc = self.qIn.get()

    def setInps(self, inpArr):
        self._inp = inpArr

    def selectInps(self):
        l1 = self.qIn.get()
        for i in range(len(self._inp)):
            if self._inp[i] != None:
                ot = OT(self.PKE)
                ot.chose(self._inp[i])
                vl = ot.runOn2(self.qIn, self.qOut)
                t = label(l1, 32)
                t.val = vl.to_bytes(32, "little")
                self.inp[i] = t

    def setAliceInp(self):
        self.inp = self.qIn.get()

    def evaluate(self):
        inp = self.inp
        for i in range(len(self.gc.circ)):
            inp = self.gc.propgate(inp, i)
        self.OUT = inp[0]
        self.out = self.gc.DecodeOutput(inp[0])

    def Run(self):
        self.setAliceInp()
        self.setCirc()
        self.selectInps()
        self.evaluate()


def changeVars(dic, lyr):
    update = lambda x: dic[x] if x >= 0 else -1
    for i in range(len(lyr)):
        lyr[i] = (lyr[i][0], update(lyr[i][1]), update(lyr[i][2]))
    return lyr


def copyOf(circ):
    circ2 = []
    for lyr in circ:
        l = []
        for node in lyr:
            l.append((node[0], node[1], node[2]))
        circ2.append(l)
    return circ2


# add two numbers in the range (0-7), and output the msb ((a+b)>>2)
sumCirc = [
    #     0           1            2           3          4          5        
    #   x0+y0      x1 + y1      x0 * y0    x1 * y1       x1        x2 + 1     
    [("mul", 0, 3), ("mul", 1, 4), ("eq", 1, 1), ("not", 2, -1), ("eq", 4, 4), ("not", 5, -1)],
    #   Out1        Out2      1 + x1*y1    x1*x0*y0    y1*x0*y0    (x2 + 1) * (1 + y2)
    [("not", 1, -1), ("mul", 2, 0), ("mul", 4, 0), ("mul", 3, 5)],
    #   Out1        Out2   1 + x1*x0*y0|1 + y1*x0*y0|   ???
    [("not", 1, -1), ("not", 2, -1), ("mul", 0, 3)],
    [("mul", 0, 1), ("eq", 2, 2)],
    [("mul", 0, 1)],
    [("not", 0, -1)]
]
Equation2 = [
    #         0             1          2            3         4
    [("add", 0, 0), ("mul", 0, 2), ("mul", 0, 3), ("mul", 1, 2), ("mul", 1, 3)]
]
s1 = copyOf(sumCirc)
s1[0] = changeVars({0: 1, 1: 2, 2: 0, 3: 0, 4: 3, 5: 4}, s1[0])
Equation2 += s1

print("genrating the ElGamal scheme...", end=" ", flush=True)
PKE0 = PKE_ElGamal(256 + 4)
print("done")

q1, q2 = queue.Queue(), queue.Queue()
start = time.time()
print("running main test")
maxN = 10
print(0, "/", maxN, end="  \r", flush=True)
for i in range(maxN):
    for x0, x1, y0, y1 in itertools.product(range(2), repeat=4):
        a = Alice(q1, q2, PKE0)
        b = Bob(q2, q1, PKE0)
        a.setCirc(Equation2, 4, 16)
        a.setInps([x0, x1, None, None])
        b.setInps([None, None, y0, y1])
        t1 = threading.Thread(target=a.Run)
        t2 = threading.Thread(target=b.Run)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        if not (((x1 * 2 + x0) * (y1 * 2 + y0) >= 4) == (0 == b.out)):
            print()
            print("fail test with inps:", (x1 * 2 + x0), (y1 * 2 + y0))
            exit(1)
    print(i + 1, "/", maxN, end="  \r", flush=True)
print("")


# tool for communiction profiling
def moveAndSum(qIn, qOut, qSum):
    while True:
        c = qIn.get()
        qSum.put(len(pickle.dumps(c)))
        qOut.put(c)


avg = 0
for x0, x1, y0, y1 in itertools.product(range(2), repeat=4):
    q1, q2 = queue.Queue(maxsize=2), queue.Queue(maxsize=2)
    q3, q4 = queue.Queue(maxsize=2), queue.Queue(maxsize=2)
    sumQ = queue.Queue(maxsize=-1)
    a = Alice(q1, q2, PKE0)
    b = Bob(q4, q3, PKE0)
    a.setCirc(Equation2, 4, 16)
    a.setInps([x0, x1, None, None])
    b.setInps([None, None, y0, y1])
    t1 = threading.Thread(target=a.Run)
    t2 = threading.Thread(target=b.Run)
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
avg /= 16
print("avg communiction in kB:", (avg // 100) / 10)
print("Execution Time is", time.time() - start, "seconds")
