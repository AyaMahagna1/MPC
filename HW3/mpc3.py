import random,itertools
import time

class BeDOZaDealer:
    def __init__(self,size):
        self.arrA = []
        self.arrB = []
        for i in range(size):
            a,b = random.randint(0,1),random.randint(0,1)
            c = a*b
            aTag0,bTag0,cTag0 = random.randint(0,1),random.randint(0,1),random.randint(0,1)
            aTag1,bTag1,cTag1 = (aTag0 + a) % 2,(bTag0 + b) % 2,(cTag0 + c) % 2
            self.arrA.append((aTag0,bTag0,cTag0))
            self.arrB.append((aTag1,bTag1,cTag1))
    def RandA(self):
        return self.arrA
    def RandB(self):
        return self.arrB

class BeDOZaAgent:
    def __init__(self,i,circ):
        self.circ = circ
        self.i = i
        self.inp = []
    def Init(self,inp,BeTrip):
        self.orginp = inp
        self.BeTrip = BeTrip
    def SendInp(self):
        msg = []
        for i in self.orginp:
            iTag = random.randint(0,1)
            self.inp.append((i+iTag)%2)
            msg.append(iTag)
        return msg
    def ReceiveInp(self,inp2):
        if self.i == 1:
            self.inp = self.inp + inp2
        elif self.i == 2:
            self.inp = inp2 + self.inp
    def Send(self):
        thisLayer = self.circ[0]
        self.circ = self.circ[1:]
        self.nextInp = []
        msg = []
        for node in thisLayer:
            x,y = self.inp[node[1]],self.inp[node[2]]
            if node[2] < 0:
                y = (self.i) % 2
            if node[1] < 0:
                x = (self.i) % 2
            if node[0] == "add":
                self.nextInp.append((x+y) % 2)
            if node[0] == "eq":
                self.nextInp.append(x)
            if node[0] == "mul":
                #mul case
                u,v,w = self.BeTrip[0]
                self.BeTrip = self.BeTrip[1:]
                d = (x+u) % 2
                e = (y+v) % 2
                self.nextInp.append((d,e,x,y,w)) #save on the node for latter calculations
                msg.append((d,e))
        return msg
    def Receive(self,msg):
        i = 0
        for node in self.nextInp:
            if type(node) == type((0,)):
                dTag0,eTag0 = msg[0]
                msg = msg[1:]
                dTag1,eTag1,x,y,w = node
                d,e = (dTag0+dTag1)%2,(eTag0+eTag1)%2
                z = (w + (e * x) + (d*y))%2
                if self.i == 1:
                    z = (z + (e*d))%2
                self.nextInp[i] = z
            i += 1
        self.inp = self.nextInp
    def SendOut(self):
        return self.inp
    def ReceiveOut(self,inp2):
        out = []
        for i in range(len(inp2)):
            out.append((self.inp[i]+inp2[i])%2)
        return out
    def ended(self):
        return len(self.circ) == 0


#add two numbers in the range (0-7) such that
#sumCirc(x,y) = x+y if x+y < 4
#and
#sumCirc(x,y) >= 4 if x+y >= 4
sumCirc = [
    #     0           1            2           3          4          5          6           7
    #   x0+y0      x1 + y1      x0 * y0    x1 * y1       x1        x2 + 1       y1        1 + y2
    [("add",0,3),("add",1,4),("mul",0,3),("mul",1,4),("eq",1,1),("add",2,-1),("eq",4,4),("add",5,-1)],
    #   Out1        Out2      1 + x1*y1    x1*x0*y0    y1*x0*y0    (x2 + 1) * (1 + y2)
    [("eq",0,0),("add",1,2),("add",3,-1),("mul",4,2),("mul",6,2),("mul",5,7)],
    #   Out1        Out2   1 + x1*x0*y0|1 + y1*x0*y0|   ???
    [("eq",0,0),("eq",1,1),("add",3,-1),("add",4,-1),("mul",2,5)],
    [("eq",0,0),("eq",1,1),("mul",2,3),("eq",4,4)],
    [("eq",0,0),("eq",1,1),("mul",2,3)],
    [("eq",0,0),("eq",1,1),("add",2,-1)]
]

#utilitys to create circuits
def changeVars(dic,lyr):
    update = lambda x:dic[x] if x >=0 else -1
    for i in range(len(lyr)):
        lyr[i] = (lyr[i][0],update(lyr[i][1]),update(lyr[i][2]))
    return lyr
def addToVars(k,lyr):
    update = lambda x:x+k if x >=0 else -1
    for i in range(len(lyr)):
        lyr[i] = (lyr[i][0],update(lyr[i][1]),update(lyr[i][2]))
    return lyr
def appendHorizontaly(appTo,circ,lvl):
    for i in range(lvl,lvl+len(circ)):
        if len(appTo) <= i:
            appTo = appTo + [[]] #adding empty layer
        lyr2 = circ[i-lvl]
        if i > lvl:
            lyr2 = addToVars(len(appTo[i-1])-len(circ[i-lvl-1]),lyr2)
        appTo[i] = appTo[i] + (lyr2)
    return appTo
def copyOf(circ):
    circ2 = []
    for lyr in circ:
        l = []
        for node in lyr:
            l.append((node[0],node[1],node[2]))
        circ2.append(l)
    return circ2
def mulSizeOfCirc(circ):
    cnt = 0
    for lyr in circ:
        for node in lyr:
            if node[0] == "mul":
                cnt += 1
    return cnt


#the main circuit
Equation3 = [
#         0             1          2            3           4          5            6           7            8
    [("add",-1,-1),("mul",0,4),("mul",0,5),("mul",1,4),("mul",1,5),("mul",2,6),("mul",2,7),("mul",3,6),("mul",3,7)]
]

s1 = copyOf(sumCirc)
s2 = copyOf(sumCirc)
s3 = copyOf(sumCirc)
s1[0] = changeVars({0:1,1:2,2:0,3:0,4:3,5:4},s1[0])
s2[0] = changeVars({0:5,1:6,2:0,3:0,4:7,5:8},s2[0])
s3[0] = changeVars({0:0,1:1,2:2,3:3,4:4,5:5},s3[0])
Equation3 = appendHorizontaly(Equation3,s1,1)
Equation3 = appendHorizontaly(Equation3,s2,1)
Equation3 = appendHorizontaly(Equation3,s3,7)
Equation3.append([("eq",2,2)])

#the runner get inputs x,y and return circ(x,y), the communication complexity
def runner(x,y,circ):
    communication = 0
    Dealer = BeDOZaDealer(mulSizeOfCirc(circ))
    Alice,Bob = BeDOZaAgent(1,circ),BeDOZaAgent(2,circ)
    Alice.Init(x, Dealer.RandA())
    Bob.Init(y, Dealer.RandB())
    msg1 = Alice.SendInp()
    msg2 = Bob.SendInp()
    communication += len(msg1)+ len(msg2)
    Bob.ReceiveInp(msg1)
    Alice.ReceiveInp(msg2)
    while(not Alice.ended()):
        msg1 = Alice.Send()
        msg2 = Bob.Send()
        communication += len(msg1)+ len(msg2)
        Bob.Receive(msg1)
        Alice.Receive(msg2)
    msg1 = Bob.SendOut()
    communication += len(msg1)
    z = Alice.ReceiveOut(msg1)
    return z,communication
start_time=time.time()
commCnt = 0
maxComm = 0
i = 0
print("running the main test 100 times on every posible input and cheak that the output is right")
for trys in range(100):
    print(trys,"/100",end="\r")
    for x1,x2,a1,a2 in itertools.product(range(0,4),repeat=4):
        x = [x1%2,x1//2,x2%2,x2//2]
        y = [a1%2,a1//2,a2%2,a2//2]
        out,cc = runner(x,y,Equation3)
        commCnt += cc
        i += 1
        if maxComm < cc:
            maxComm = cc
        if (out[0] == 1) != (a1*x1+a2*x2>=4):
            print(a1,a2,x1,x2)
print("\n")
print("the communication complexity is:")
print("avg:",commCnt/i,"\nw.c:",maxComm)
print("---Execution time is: %s seconds ---" % (time.time() - start_time))

