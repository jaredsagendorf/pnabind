import numpy as np

def signedVolume(a, b, c, d):
    """Computes the signed volume of a series of tetrahedrons defined by the vertices in 
    a, b c and d. The ouput is an SxT array which gives the signed volume of the tetrahedron defined
    by the line segment 's' and two vertices of the triangle 't'."""
    
    return np.sum((a-d)*np.cross(b-d, c-d), axis=2)

def segmentsIntersectTriangles(s, t):
    """For each line segment in 's', this function computes whether it intersects any of the triangles
    given in 't'."""
    # compute the normals to each triangle
    normals = np.cross(t[2]-t[0], t[2]-t[1])
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    
    # get sign of each segment endpoint, if the sign changes then we know this segment crosses the
    # plane which contains a triangle. If the value is zero the endpoint of the segment lies on the 
    # plane.
    # s[i][:, np.newaxis] - t[j] -> S x T x 3 array
    sign1 = np.sign(np.sum(normals*(s[0][:, np.newaxis] - t[2]), axis=2)) # S x T
    sign2 = np.sign(np.sum(normals*(s[1][:, np.newaxis] - t[2]), axis=2)) # S x T
    
    # determine segments which cross the plane of a triangle. 1 if the sign of the end points of s is 
    # different AND one of end points of s is not a vertex of t
    cross = (sign1 != sign2)*(sign1 != 0)*(sign2 != 0) # S x T 
    
    # get signed volumes
    v1 = np.sign(signedVolume(t[0], t[1], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T
    v2 = np.sign(signedVolume(t[1], t[2], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T
    v3 = np.sign(signedVolume(t[2], t[0], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T
    
    same_volume = np.logical_and((v1 == v2), (v2 == v3)) # 1 if s and t have same sign in v1, v2 and v3
    
    return np.nonzero(np.logical_not(np.sum(cross*same_volume, axis=1)))[0]
