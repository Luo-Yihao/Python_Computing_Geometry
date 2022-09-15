import numpy as np
import pyvista as pv
from scipy.spatial import distance
def clearMesh_easy(mesh):

    
    # reducing index for faces
    Faces = []
    i = 0
    tem = mesh.faces
    while len(tem)>0:
        n_tem = tem[0] 
        Faces.append(tem[1:1+n_tem])
        tem = tem[1+n_tem:]

    return mesh.points, Faces

def clearMesh(mesh):

    def root(d,a):
        if a not in d:
            return a 
        else:
            return root(d,d[a])
    
    # removing repeated points
    dist0 = np.triu(distance.cdist(mesh.points, mesh.points, 'euclidean'))
    a,b = np.where(dist0==0)

    points = {k:mesh.points[k] for k in range(len(mesh.points))}

    c = np.where(b-a>0)[0]

    d = {}
    for i in c:
        if b[i] in d:   
            continue      
        d[b[i]] = a[i]
        del points[b[i]]
        
    dd = {k:root(d,k) for k in range(mesh.n_points)}
    ee = list(set(dd.values())) 
    
    ff = {ee[i]:i for i in range(len(ee))}
    hh = {k:ff[dd[k]] for k in range(mesh.n_points)}
    points = {hh[k]:points[k] for k in points.keys()}
    
    
    
    # reducing index for faces
    Faces = []
    i = 0
    tem = mesh.faces
    while len(tem)>0:
        n_tem = tem[0] 
        tem_shape = tem[1:1+n_tem]
        tem_shape = [hh[ele] for ele in tem_shape]
        Faces.append(tem_shape)
        tem = tem[1+n_tem:]

    return points, Faces