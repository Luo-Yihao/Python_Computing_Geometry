from manifold import *
from topologyProcess import *
import numpy as np
from scipy.sparse import *
from scipy.sparse import linalg

def HodgeDecompose(m2,gamma, onlyH = False):
    c2 = m2.toChain()
    areaVec = m2.areaTri()
        
    delta1 = c2.partial1.multiply(c2.cotWight)
    L0 = delta1.dot(c2.partial1.transpose())

    delta2T = c2.partial2.multiply(1/areaVec).transpose().multiply(1/c2.cotWight)
    delta2 = delta2T.transpose()
    L2 = delta2T.dot(c2.partial2).transpose()

    deltagamma = delta1.dot(gamma)
    P = linalg.spsolve(L0,deltagamma)
    dP = c2.partial1.transpose().dot(P)

    dgamma = c2.partial2.transpose().dot(gamma)
    Q = linalg.spsolve(L2,dgamma)
    codQ = delta2.dot(Q)

    H = gamma - dP - codQ
    
    if onlyH == True:
        return H

    return H, dP, codQ 


def HodgeBase_dim1(m2):
    c2 = m2.toChain()
    basesloop = fundamentalGroup(m2)
    cobases = [toCochain(c2, loop) for loop in basesloop] 

    cuttrace = set()
    for loop in basesloop:
        cuttrace = cuttrace.union(loop)

    Hbases = [HodgeDecompose(m2, coloop ,onlyH= True) for coloop in cobases]

    intH = [intergalCochain(H,m2,cuttrace) for H in Hbases]

    return Hbases, intH
    

def conformalModel_circleStripe(m2: Manifold2D):
    c2 = m2.toChain()
    areaVec = m2.areaTri()
    G1,G2 = m2.toGraph()
    b = m2.boundary()

    if c2.EularChar != 0 or len(b) != 2:
        return 'Input surface is not a topological circleStripe'
    
    delta1 = c2.partial1.multiply(c2.cotWight)
    L0 = delta1.dot(c2.partial1.transpose())

    delta2T = c2.partial2.multiply(1/areaVec).transpose().multiply(1/c2.cotWight)
    delta2 = delta2T.transpose()
    L2 = delta2T.dot(c2.partial2).transpose()

    boundary = [[c2.toEdge(HE)[0]*c2.toEdge(HE)[1] for HE in ele] for ele in b]
    boundaryPoint = [[HE.startP.ID for HE in ele] for ele in b]

    f = np.zeros((c2.N_0,1))
    
    blen = [len(ele) for ele in b]

    # bounded condition 
    b0 = blen.index(max(blen))
    b1 = blen.index(min(blen))

    f[boundaryPoint[b0]] = 1
    f[boundaryPoint[b1]] = 0

    bdcondition = L0[:,boundaryPoint[b0]].dot(f[boundaryPoint[b0]])
    bdcondition += L0[:,boundaryPoint[b1]].dot(f[boundaryPoint[b1]])


    innerPoint = set(range(c2.N_0)).difference(boundaryPoint[0],boundaryPoint[1])
    innerPoint = list(innerPoint)
    
    # solving Laplacian equation  
    f[innerPoint] = linalg.lsqr(L0[:,innerPoint],-bdcondition)[0].reshape(-1,1)


    # search minimal cut line 
    innerb = [ele.startP.ID for ele in b[b1]]
    outb = [ele.startP.ID for ele in b[b0]]
    dist2, path2 = gh.shortest_path(G1,indices=innerb, directed=False,return_predecessors=True)

    dtem = dist2[:,outb]
    aa = dtem.argmax()//dtem.shape[1]
    bb = dtem.argmax()%dtem.shape[1]
    fromP = innerb[aa]
    toP = outb[bb]

    shortpath = [toP]
    cutline = []
    while shortpath[0] != fromP:
        temPtID = path2[aa,shortpath[0]]
        
        cutline = list(m2.Vert_set[temPtID].startfrom.intersection(m2.Vert_set[shortpath[0]].endby))+cutline

        shortpath = [temPtID] + shortpath


    shortpathPos = np.array([m2.Vert_set[pt].pos for pt in shortpath])

    coT = toCochain(c2, cutline)

    H = HodgeDecompose(m2,coT, onlyH=True)

    return f.transpose()[0], intergalCochain(H,m2,cutline)
