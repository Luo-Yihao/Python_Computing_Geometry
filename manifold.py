from tokenize import Double
import numpy as np
import pyvista as pv
from scipy.sparse import *

class chainComplex:
    def __init__(self,partials):
        self.dim = len(partials)
        self.N_0 = partials[0].shape[0]
        self.EularChar = self.N_0
        self.Edge = None
        self.cotWight = None
        for i in range(len(partials)):
            exec('self.partial'+str(i+1)+' = partials['+str(i)+']')
            exec('self.N_'+str(i+1)+' = partials['+str(i)+'].shape[1]')
            self.EularChar += (-1)**(i+1)*partials[i].shape[1]

    def Laplacian(self,k=1):
        P1 = eval('self.partial'+str(k))
        P2 = eval('self.partial'+str(k+1))
        L = P1.transpose().dot(P1)+P2.dot(P2.transpose())
        return L

    def EdgeDoc(self):
        if self.Edge != None:
            return self.Edge
        Edge = {}
        csrtem = self.partial1.tocsc()
        for j in range(csrtem.shape[1]):
            edgetem = find(csrtem.getcol(j))

            if edgetem[2][0]==1:
                edgetem = edgetem[0][::-1].tolist()
            else:
                edgetem = edgetem[0].tolist()
            Edge[tuple(edgetem)] = j
        self.Edge = Edge
        return Edge

    def toEdge(self,HE):
        if self.Edge==None:
            self.EdgeDoc()
        try: 
            N = self.Edge[(HE.startP.ID, HE.endP.ID)] 
            return [N,1]
        except:
            N = self.Edge[(HE.endP.ID, HE.startP.ID)] 
            return [N,-1]
    

class Vertex:
    def __init__(self, pos):
        self.pos = pos
        self.startfrom = set()
        self.endby = set()
        self.Next = set()
        self.ID = None

class HE_edge:
    global start0
    global end0
    def __init__(self, start0, end0):
        self.startP = start0
        self.endP = end0
        start0.Next.add(end0)
        start0.startfrom.add(self)
        end0.endby.add(self)

        self.inner = set()

        self.Next = set()

        self.revHE = None

    def reversed(self):
        if self.revHE== None:
            temHE = HE_edge(self.endP,self.startP)
            self.revHE = temHE                    
            self.revHE.revHE = self
        return self.revHE
    def connect(self, next):
        if next.startP == self.endP:
            self.next = next
    def tovector(self):
        return self.endP.pos - self.startP.pos

class Face:
    global HE_set
    def __init__(self, HE_set):
        self.HE_set = HE_set
        self.n_HE = len(HE_set)
        for HE in HE_set:
            HE.inner.add(self)

        self.nadjoint = set()
        self.padjoint = set()
        self.revFace = None
        self.ort = None
        self.ID = None

        for HE in HE_set:
            HE.Next = HE.Next.union(HE_set.difference({HE}))


    def reversed(self):
        if self.revFace == None:
            self.revFace = Face({ele.reversed() for ele in self.HE_set})
            self.revFace.revFace =self
            if self.ort !=None:
                self.revFace.ort = -self.ort
        return self.revFace
        
    global next
    def pconnect(self, next):
        if len(next.HE_set.intersection({ele.revHE for ele in self.HE_set}))>0:
            self.padjoint.add(next)
            next.padjoint.add(self)
        else:
            print('wrong!')
        
    def nconnect(self, next):
        if len(next.HE_set.intersection(self.HE_set))>0:
            self.nadjoint.add(next)
            next.nadjoint.add(self)
        else:
            print('wrong!')

class Manifold2D:
    def __init__(self, Points, Faces):
        self.Vert_set = {}
        self.HE_set = set()
        self.Face_set = {}
        self.Isorientable = None
        self.cotWight = None
        for k in range(len(Points)):
            self.Vert_set[k] = Vertex(Points[k])
            self.Vert_set[k].ID = k

        for k in range(len(Faces)):
            
            
            temHEs = set()
            Facealongs = set() #与新生成面正相邻的面
            Faceinvs = set() #与新生成面逆相邻的面
            

            ploygon = Faces[k]
            for i in range(len(ploygon)):
                #可能的以占用边
                HEalong = set()
                HEinv = set()
                #当前边
                starttem = self.Vert_set[ploygon[i]]
                try: 
                    endtem = self.Vert_set[ploygon[i+1]]
                except:
                    endtem = self.Vert_set[ploygon[0]]

                #当前边是否已经生成
                HEalong = HEalong.union(starttem.startfrom.intersection(endtem.endby))
                if len(HEalong) >0:
                    temHE = list(HEalong)[0]  
                    # 若已经存在则存当前边之逆
                    self.HE_set.add(temHE.reversed())  
                    # print([ele.revHE in self.HE_set for ele in self.HE_set])
                    #存当前边的内面为邻面
                    Facealongs = Facealongs.union(temHE.inner)
                        
                else:
                    #当前边的逆是否已经生成
                    HEinv = HEinv.union(starttem.endby.intersection(endtem.startfrom))
                    if len(HEinv)>0:
                        #存当前边的逆的内面为逆边邻面
                        Faceinvs = Faceinvs.union(list(HEinv)[0].inner)
                        temHE = list(HEinv)[0].reversed()
                        
                    #新边
                    else:
                        temHE = HE_edge(starttem,endtem)

                
                   #存当前边
                    self.HE_set.add(temHE)
                
                temHEs.add(temHE)    

            temFace = Face(temHEs)
            temFace.ID = k
            
            for ele in Faceinvs:
                temFace.pconnect(ele)
                    
            
            for ele in Facealongs:
                temFace.nconnect(ele)
            
            self.Face_set[k] = temFace

        print(self.orientate())
                    
    def turnover(self):
        if self.Isorientable == False:
            return 'unorientable' 
        for k in range(len(self.Face_set)):
            self.Face_set[k].ort = -self.Face_set[k].ort
        

    def orientate(self, ploygonID=0, outside=True):
        try:
            ploygon = self.Face_set[ploygonID]
        except:
            return 'null'

        if self.Isorientable == True:
            if (outside==False and ploygon.ort==1) or (outside==True and ploygon.ort==0):
                self.turnover()
            return 'orientated' 
        elif self.Isorientable == False:
            return 'unorientable' 


        
        ploygon = self.Face_set[ploygonID]

        if outside:
            pskark = [ploygon.ID]
            nskark = [None]
        else:
            nskark = [ploygon.ID]
            pskark = [None]



        i = 0
        while i<=len(pskark) or i<=len(nskark) :


            if i<len(pskark):
                ptem = pskark[i]
                if ptem != None:

                    ptem = self.Face_set[pskark[i]]
                    ptem.ort = 1

                    for ele in ptem.padjoint:
                        if ele.ort == None:
                            self.Face_set[ele.ID].ort = 2
                            pskark.append(ele.ID)
                        elif ele.ort < 0:
                            self.Face_set[ele.ID].ort = 0 
                            self.Isorientable = False
                            return "unorientable"

                    for ele in ptem.nadjoint:
                        if ele.ort == None:
                            self.Face_set[ele.ID].ort = -2
                            nskark.append(ele.ID)
                        elif ele.ort > 0:
                            self.Face_set[ele.ID].ort = 0 
                            self.Isorientable = False
                            return "unorientable"

            if i<len(nskark):
                ntem = nskark[i]

                if ntem != None:
                    
                    ntem = self.Face_set[ntem]
                    ntem.ort = -1

                    for ele in ntem.nadjoint:
                        if ele.ort == None:
                            self.Face_set[ele.ID].ort = 2
                            pskark.append(ele.ID)
                        elif ele.ort < 0:
                            self.Face_set[ele.ID].ort = 0 
                            self.Isorientable = False
                            return "unorientable"

                    for ele in ntem.padjoint:
                        if ele.ort == None:
                            self.Face_set[ele.ID].ort = -2
                            nskark.append(ele.ID)
                        elif ele.ort > 0:
                            self.Face_set[ele.ID].ort = 0 
                            self.Isorientable = False
                            return "unorientable"          
            
            i += 1
        
        self.Isorientable = True
        return "orientated"
    
    def boundary(self):
        boundpos = self.HE_set.difference({ele.revHE for ele in self.HE_set})
        bounds = []
        try:
            bounds.append([boundpos.pop()])
        except:
            print('closed manifold without boundary')
            return []
            
    
        
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
        
        if self.Isorientable == True:
            for i in range(len(bounds)):
                tembound = bounds[i]
                if list(tembound[0].inner)[0].ort == -1:
                    ntem = len(tembound)
                    bounds[i] = [tembound[tem-j].reversed() for j in range(ntem)]

        return bounds

    def toMesh(self):
        PP = []
        for i in range(len(self.Vert_set)):
            PP.append(self.Vert_set[i].pos)

        FF = []
        for ele in self.Face_set.values():
            temHEs = {ele for ele in ele.HE_set}
            HE = temHEs.pop()

            F = [len(temHEs)+1,HE.startP.ID]
            

            while len(temHEs)>0:
                HE = HE.endP.startfrom.intersection(temHEs).pop()
                F.append(HE.startP.ID)
                temHEs.remove(HE)       
            
            FF.append(F)

        return pv.PolyData(np.array(PP),np.hstack(FF))

    def toGraph(self):
        # initialize the PoincareDual Graph 
        n = len(self.Vert_set)
        m = len(self.Face_set)

        rowV = []
        rowF = []
        colV = []
        colF = []

        PoincareDual = {}
        for HE in self.HE_set:

            rowV.append(HE.startP.ID)
            colV.append(HE.endP.ID)
            
            if len(HE.inner)==2:
                face0 = list(HE.inner)[0]
                face1 = list(HE.inner)[1]
                rowF += [face1.ID,face0.ID]
                colF += [face0.ID,face1.ID]

            elif len(HE.inner)==1:
                face0 = list(HE.inner)[0]
                try:
                    face1 = list(HE.revHE.inner)[0]
                    rowF.append(face0.ID)
                    colF.append(face1.ID)
                except:
                    continue


        rowV = np.array(rowV)
        colV = np.array(colV)
        rowF = np.array(rowF)
        colF = np.array(colF)
        dataV = np.ones(len(colV))
        dataF = np.ones(len(colF))
        adjmatrix = coo_matrix((dataV,(rowV,colV)),shape=(n,n))
        PoincareDual = coo_matrix((dataF,(rowF,colF)),shape=(m,m))

        adjmatrix = adjmatrix.tolil()
        PoincareDual = PoincareDual.tolil()
        return adjmatrix, PoincareDual


    def toChain(self):
        try:
            return self.chain
        except:
            Edge = set()
            


        bl = len(self.HE_set.difference({he.revHE for he in self.HE_set}))
        n1 = int((len(self.HE_set) - bl)/2 + bl)
        partial1 = dok_matrix(( len(self.Vert_set),n1))
        partial2 = dok_matrix((n1,len(self.Face_set)))

        EdgeOut = {}
        for he in self.HE_set:
            if he.revHE in Edge:
                continue

            temid = len(Edge)
            Edge.add(he) 
            EdgeOut[(he.startP.ID,he.endP.ID)] = temid
            partial1[he.startP.ID,temid] = -1 
            partial1[he.endP.ID,temid] = 1
            for facetem in he.inner:
                partial2[temid,facetem.ID] = 1
            if he.revHE == None:
                continue 
            for facetem in he.revHE.inner:
                partial2[temid,facetem.ID] = -1
        
        C2 = chainComplex([partial1, partial2])
        C2.Edge = EdgeOut
        C2.cotWight = self.cotWight_vector(EdgeOut)
        self.chain = C2 
        
        return C2

    def cotWight_vector(self,EdgeOut):
        #print(EdgeOut)
        cotWight = np.zeros(len(EdgeOut))
            
        for HE in EdgeOut.keys():
            temHE = self.Vert_set[HE[0]].startfrom.intersection(self.Vert_set[HE[1]].endby).pop()
            wtem = 0
            try:
                Fs = temHE.inner.union(temHE.revHE.inner)
            except:
                Fs = temHE.inner
            for F in Fs:
                HEsides = list(F.HE_set.difference({temHE,temHE.revHE}))

                Aside = HEsides[0].tovector()
                Bside = -HEsides[1].tovector()

                costem2 = (np.inner(Aside,Bside))**2/(np.inner(Aside,Aside)*np.inner(Bside,Bside))
                wtem += np.sqrt(costem2/(1 - costem2))/2

            cotWight[EdgeOut[HE]] = wtem

        return cotWight
    
    def areaTri(self):
        try:
            return self.areaVec
        except:
            areaVec = np.empty((len(self.Face_set),))
        for F in self.Face_set.values():
            psTem = [HE.startP.pos for HE in F.HE_set]
            triTem = np.ones((3,3))
            triTem[1,:] = psTem[1]-psTem[0]
            triTem[2,:] = psTem[2]-psTem[0]
            areaVec[F.ID] = 1/2*np.linalg.det(triTem)
        self.areaVec = areaVec
        return self.areaVec


