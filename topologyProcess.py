from importlib.resources import path
from textwrap import shorten
import numpy as np 
import scipy.sparse.csgraph as gh
from scipy.sparse import *

def fundamentalGroup(m2,needshorten=False):

    # initialize the PoincareDual Graph 
    adjmatrix, PoincareDual = m2.toGraph()

    # gTree of PoincareDual is the Basic Area D2 
    gTree = gh.breadth_first_tree(PoincareDual,i_start=0,directed=False)
    gTree = gTree + gTree.transpose()
    gTree = gTree.tolil()

    # construct the cutGraph 
    cutGraph = adjmatrix.copy()
    for ele in gTree.todok().keys():
        face0 = m2.Face_set[ele[0]]
        face1 = m2.Face_set[ele[1]]
        if face1 in face0.padjoint:
            HEconnect = face0.HE_set.intersection({HE.revHE for HE in face1.HE_set}).pop()
        else:
            HEconnect = face0.HE_set.intersection(face1.HE_set).pop()
        cutGraph[(HEconnect.startP.ID,HEconnect.endP.ID)] = 0 
        cutGraph[(HEconnect.endP.ID,HEconnect.startP.ID)] = 0 
    
    # shortengraph: To cut open branches off  
    def shortengraph(G):
        cutedG = G.copy()
        while 1:
            degree = np.array([len(ele) for ele in cutedG.rows])
            leaves= np.where(degree==1)[0] 
            if len(leaves)==0:
                return cutedG
            #print(len(leaves))
            for i in leaves:
                j = set(cutedG[i].todok().keys()).pop()[1]
                cutedG[(i,j)] = 0
                cutedG[(j,i)] = 0

    truthcutgraph = shortengraph(cutGraph)
    sumbent = cutGraph - truthcutgraph

    for ele in sumbent.todok().keys():
        temHE = m2.Vert_set[ele[0]].startfrom.intersection(m2.Vert_set[ele[1]].endby)
        try:
            temHE = list(temHE)[0]
        except:
            temHE = m2.Vert_set[ele[1]].startfrom.intersection(m2.Vert_set[ele[0]].endby)
            temHE = list(temHE)[0]
        fs = list(temHE.inner)
        if len(fs)>1:
            i = fs[0].ID
            j = fs[1].ID
        elif len(fs)==1:
            i = fs[0].ID
            try:
                j = list(temHE.revHE.inner)[0].ID
            except:
                continue
        else:
            continue

        gTree[(i,j)] = 1
        gTree[(j,i)] = 1

    cutTree = gh.breadth_first_tree(cutGraph,0,directed=False)

    for ele in cutTree.todok().keys():
        #print(cutGraph[ele])
        cutGraph[ele] = 0
        cutGraph[tuple(reversed(ele))] = 0 

    bases = []

    loops = set()

    for ele in cutGraph.todok().keys():
        if tuple(reversed(ele)) in loops:
            continue
        else:
            loops.add(ele)


        HE1 = m2.Vert_set[ele[0]].startfrom.intersection(m2.Vert_set[ele[1]].endby).pop()

        f1 = list(HE1.inner)[0].ID
        f2 = list(HE1.revHE.inner)[0].ID
        dist2, path2 = gh.shortest_path(gTree,indices=[f1], directed=False,return_predecessors=True)
        ftemID = f2
        boundpos = {ele for ele in m2.Face_set[ftemID].HE_set}
        #link = [HE1.startP.pos]
        while ftemID != f1:
            ftemID1 = path2[0][ftemID]
            for ele in m2.Face_set[ftemID1].HE_set:
                if ele in boundpos or ele.revHE in boundpos:
                    boundpos.difference_update({ele, ele.revHE})
                else:
                    boundpos.add(ele)
                    
            ftemID = ftemID1
        
        bounds = [[boundpos.pop()]]
        while 1:
            start0 = bounds[-1][-1].startP
            end1 = bounds[-1][-1].endP
            while end1 != start0:
                start1 = end1

                for HE in start1.startfrom:
                    if HE in boundpos:
                        boundpos.remove(HE)
                        bounds[-1].append(HE)
                        end1 = HE.endP
                        break
                else:               
                    for HE in start1.endby:
                        if HE in boundpos:
                            boundpos.remove(HE)
                            bounds[-1].append(HE.reversed())
                            end1 = HE.startP
                            break
                    else:
                        print('something wrong')
                        return bounds, boundpos
                
                

            if len(boundpos)>0:
                bounds.append([boundpos.pop()])
            else:
                break
        

        looplens = [len(ele) for ele in bounds]
        shortindex = looplens.index(min(looplens))
    
        loopbase = bounds[shortindex]

        if needshorten:
            bases.append(shortenLoop_0(loopbase))
        else:
            bases.append(loopbase)

        

        # link.append(HE1.startP.pos)
        # link = np.array(link)
        # bases.append(link)
    
    return bases



def shortenLoop_0(Loop0):
    Loop = [ala for ala in Loop0] 
    length1 = len(Loop)
    length0 = length1 +1 
    while length1 < length0:
        length0 = length1
        j = 0
        while j < length1:
            #print('j='+str(j))
            HE0 = Loop[j]
            HE1 = Loop[(j+1)%length1]

            F1 = HE0.inner.intersection(HE1.inner)
            F2 = HE0.inner.intersection(HE1.revHE.inner)
            F3 = HE0.revHE.inner.intersection(HE1.revHE.inner)
            F4 = HE0.revHE.inner.intersection(HE1.inner)

            FF = F1.union(F2,F3,F4)
            for Facetem in FF:
                
                t = 2
                ProHEs = Facetem.HE_set.union({ele.revHE for ele in Facetem.HE_set})
                while Loop[(j+t)%length1] in ProHEs:
                    t += 1
                
                #print('t='+str(t)+',l='+str(len(Facetem.HE_set)))
                try:
                    ProHEs.difference_update(set(Loop[j:j+t]))
                except:
                    ProHEs.difference_update(set(Loop[j:]),set(Loop[:(j+t)%length1]))


                RePHE = []
                if  len(Facetem.HE_set)-t<t:
                    start1 = HE0.startP
                    end1 = Loop[(j+t-1)%length1].endP
                    while start1 != end1:
                        for HE in start1.startfrom.intersection(ProHEs):
                            ProHEs.remove(HE)
                            RePHE.append(HE)
                            start1 = HE.endP
                    if j+t<length1:
                        Loop = Loop[:j]+RePHE+Loop[j+t:]
                    else:
                        Loop = Loop[(j+t)%length1:j]+RePHE
                    length1 = len(Loop)
                    break
            else:
                j +=1
    print('function limited')
    return Loop
    




################################
def cohomolpogyBases(m2):
    basesloop = fundamentalGroup(m2)
    C2 = m2.toChain()
    coBases = []
    for loop in basesloop:
        coBases.append(coloop(C2, loop))
    return coBases, basesloop


def coloop(C2, basetem):
    coBase0 = lil_array((1,len(C2.Edge)))
    Psettem = [HEtem.startP for HEtem in basetem]
    n = len(Psettem)
    for i in range(n):
        currentem = Psettem[i]
        pretem = Psettem[(i-1)%n]
        posttem = Psettem[(i+1)%n]

        pospPoint = {HE.endP for HE in currentem.startfrom}

        pospPoint.difference_update({pretem,posttem})
        for F in basetem[i].inner.union(basetem[i].revHE.inner):
            if (F.ort == 1)==(basetem[i] in F.HE_set):

                postiveVert = list({HE.startP for HE in F.HE_set}.intersection(pospPoint))
                break
        pospPoint.difference_update(postiveVert)

        k = 0
        while k<len(postiveVert):
            # newPoint = {HE.endP for HE in postiveVert[-1].startfrom}.union({HE.startP for HE in postiveVert[-1].endby})
            # newPoint.intersection_update(pospPoint)
            HE0 = postiveVert[k].endby.intersection(currentem.startfrom).pop()
            newPoint = []
            for vtem in pospPoint:
                HE1 = vtem.endby.intersection(currentem.startfrom).pop()
                Ftems = (HE0.inner.union(HE0.revHE.inner)).intersection(HE1.inner.union(HE1.revHE.inner))
                if len(Ftems)>0:
                    newPoint.append(vtem)

            postiveVert += newPoint
            pospPoint.difference_update(newPoint)
            k += 1

        for v in postiveVert:
            try:
                N = C2.Edge[(currentem.ID,v.ID)]
                coBase0[0,N] = 1
            except:
                N = C2.Edge[(v.ID,currentem.ID)]
                coBase0[0,N] = -1
                    
    return coBase0     

def toCochain(C2, basetem):
    # cut C2 along basetem, and return its relevant coChain vanishing on basetem

    coChain = np.zeros((C2.N_1,))
    Psettem = {HEtem.startP for HEtem in basetem}
    Psettem.add(basetem[-1].endP)

    

    pospHE = set()

    for pointtem in Psettem:
        #pospPoint = pospPoint.union(pointtem.Next)
        pospHE = pospHE.union(pointtem.startfrom,pointtem.endby)
    
    pospHE.difference_update(basetem)
    pospHE.difference_update({bhe.revHE for bhe in basetem})
        

    postiveHE = [pospHE.pop()]

    i = 0

    while i<len(postiveHE):
        pospHE.difference_update({postiveHE[i],postiveHE[i].reversed()})
        postiveHE += list(postiveHE[i].Next.intersection(pospHE))
        postiveHE += list(postiveHE[i].revHE.Next.intersection(pospHE))
        i += 1 
    
    for HEtem in postiveHE:
        Ind = C2.toEdge(HEtem)
        coChain[Ind[0]] = Ind[1] 
                    
    return coChain  


def intergalCochain(H0,m2,cuttrace):
    interKeys = {0}
    c2 = m2.toChain()
    interValues = [[0,0]]
    interH0 = np.empty(c2.N_0)
    i = 0
    while i<len(interKeys):
        temN = interValues[i][0]
        temV = interValues[i][1]
        for HE in m2.Vert_set[temN].startfrom:
            if HE in cuttrace or HE.revHE in cuttrace:
                continue 
            if HE.endP.ID in interKeys:
                continue
            else:
                interKeys.add(HE.endP.ID)
                try:
                    newV = temV + H0[c2.Edge[(temN,HE.endP.ID)]]
                except:
                    newV = temV - H0[c2.Edge[(HE.endP.ID,temN)]]

                interValues.append([HE.endP.ID,newV])
                interH0[HE.endP.ID] = newV
        i +=1 

    return interH0 - interH0.min()

