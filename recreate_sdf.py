import numpy as np
import pickle

"""
CREATION OF SDF - COPIED FROM TASK 3 - CHANGED TO 200 SUBDIVISIONS
"""
def recreate():
    X, Y, Z = np.mgrid[-1:1:200j, -1:1:200j, -1:1:200j]
    XYZ = (X, Y, Z)
    o = np.zeros(3)
    
    # Element wise norm
    def length(X, Y, Z):
        return np.sqrt(X**2 + Y**2 + Z**2)
    
    def recenter(X, Y, Z, c):
        return (X - c[0], Y - c[1], Z - c[2])
    
    def sphere(XYZ, r, c):
        """
        r : radius of sphere, float
        c : center of sphere, 3-tuple
        """
        XYZuse = recenter(*XYZ, c)
        return length(*XYZuse) - r
    
    # xyz is one coordinate - this is working properly
    def rect(xyz, d, c):
        xyz = np.abs(xyz) - d
        return length(*np.maximum(xyz, o)) + np.amin(np.minimum(np.maximum(xyz[0],np.maximum(xyz[1],xyz[2])), o))
    
    def rectangle(XYZ, d, c):
        """
        d : dimensions of rectangle, 3-tuple
        c : center of rectangle, 3-tuple
        """
        XYZuse = recenter(*XYZ, c)
        X, Y, Z = XYZuse
        A = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)])
        B = np.zeros(200**3)
        for i in range(200**3):
            B[i] = rect(A[:,i], d, c)
            
        phi = B.reshape(200, 200, 200)
        return phi

    def cylinder(XYZ, r, c, direc):
        XYZuse = recenter(*XYZ, c)
        X, Y, Z = XYZuse
        if direc == 'z':
            return np.sqrt(X**2 + Y**2) - r
        if direc == 'y':
            return np.sqrt(X**2 + Z**2) - r
        if direc == 'x':
            return np.sqrt(Y**2 + Z**2) - r
    
    # Sphere of radius .4 centered at (.5, .5, .5)
    #s = sphere(XYZ, .4, (.5,.5,.5))
    #s = sphere(XYZ, 1.0, (.0, .0, .0))
    
    # Two spheres
    #s1 = sphere(XYZ, .25, (.4,.6,.5))
    #s2 = sphere(XYZ, .25, (.6,.4,.5))
    #s  = np.minimum(s1, s2)
    
    # Rectangle of dimension .4 x .2 x .1 centered at (.5, .5, .5)
    #s = rectangle(XYZ, (.4,.2,.1), (.5,.5,.5))
    
    # Infinite cylinder of radius .1 centered at (.5, .5, .5)
    #s = cylinder(XYZ, .1, (.5,.5,.5), 'z')
    
    # Sphere cut by three smaller spheres
    if False:
        s  = sphere(XYZ, .3, (.5,.5,.5))
        s1 = sphere(XYZ, .1, (.2,.5,.5))
        s2 = sphere(XYZ, .1, (.5,.2,.5))
        s3 = sphere(XYZ, .1, (.5,.5,.2))
        c  = np.minimum(np.minimum(s1, s2), s3)
        s  = np.maximum(s, -c)
    
    with open('sdf_trunc_cylinder.dictionary', 'wb') as dictionary_file:
        pickle.dump(s, dictionary_file)
    
    # do not return anything - the new SDF is written as a dictionary file that will be read later