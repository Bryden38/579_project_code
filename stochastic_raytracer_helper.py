# Helper functions for stochastic raytracer

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn import preprocessing

# Create a linear interpolant in each of the x, y, z directions
def create_interpolant(xyz, g):
    fx = interpolate.RegularGridInterpolator(xyz, g[0])
    fy = interpolate.RegularGridInterpolator(xyz, g[1])
    fz = interpolate.RegularGridInterpolator(xyz, g[2])
    return (fx, fy, fz)

#
def get_sdf_components(dict_name, xyz, space_subdivs):
    with open(dict_name, 'rb') as dictionary_file:
        r = pickle.load(dictionary_file)

    # Compute interpolant of the sdf
    phi = interpolate.RegularGridInterpolator(xyz, r)
        
    # Compute gradient (2/space_subdivs corresponds to the step size since r is space_subdivs subdivisions over an interval of 2)
    g = np.gradient(r, 2/space_subdivs)
    
    # Interpolant of gradient in each of x,y,z directions
    fx, fy, fz  = create_interpolant(xyz, g)
    
    return (phi, fx, fy, fz)

# Closest point operator
def cp(points, phi, fx, fy, fz):
    """
    points - a point in R^3 as a 3 ndarray, or multiple points as a 3xN ndarray
    phi    - interpolant of the sdf given by the array
    fu     - interpolant of gradient in the direction of u for u in [x,y,z]
    
    Returns the closest point on the level set 0
    """
    phi_points  = phi(points).reshape(len(points), 1)       # reshaping to (N, 1) for broadcasting with points_grad
    px, py, pz  = fx(points), fy(points), fz(points)        # the gradient of the generated points in each direction
    points_grad = np.transpose(np.array([px, py, pz]))
    closest     = points - phi_points * points_grad
    return closest

# Find the points traced on the plane when tracing back to camera
def get_points_on_plane(cam, closest, k, n, D):
    """
    See report for justification as to why this works!
    
    cam     - location of camera
    closest - array of the closest points
    k       - vector normal to plane
    n       - normalized vector normal to plane
    D       - scalar parameter of plane
    """
    b = preprocessing.normalize(cam - closest)    # Normalized vectors in direction of closest points to camera
    cos = b @ (-n)                                # Cosine of the angle between n (in other direction) and b, simply the dot product since they are unit vectors
    d_plane = (closest @ k + D)/np.linalg.norm(k) # Distance from point to plane that is the screen (using standard point to plane formula)
    scale   = (d_plane/cos).reshape(-1,1)         # Scale to find the vector from point to plane via vector to camera
    points_plane = closest + scale * b            # Points on the 
    return points_plane, scale

# Rotate the screen such that the camera vector is along x, then remove the x dimension
def get_rotation_matrix(k, cam):
    """
    Note: This only works if the plane passes through the z-axis. It can be generalized be changing the value of u.
    
    k            - vector normal to plane
    cam          - position of camera
    
    Returns A_mat, the inverse of T_mat, and T_mat, which brings the camera to the x-axis.
    """
    d_cam = np.linalg.norm(cam)           # Distance from origin to camera
    D_alt = - k @ cam                     # Alternate D that create the plane Ax + By + Cz + D_alt = 0 and cam lies on the plane
    u     = np.array([0, 0, -D_alt/k[2]]) # A point on this new plane (see notes above)
    v     = u - cam                       # A vector perpendicular to cam
    beta  = v*(d_cam/np.linalg.norm(v))   # Renormalize to have magnitude of cam
    gamma = np.cross(cam, beta)/d_cam     # Another vector parallel to both cam and beta with magnitude of cam
    
    A_mat    = np.transpose(np.array([cam, beta, gamma]))/d_cam # The matrix A which transform the standard coordinates with length d_cam to (cam, beta, gamma)
    T_mat    = np.linalg.inv(A_mat)                             # The transform T as a matrix
    return (A_mat, T_mat)

# Perform the rotation
def rotate_and_kill_x(T_mat, points_plane):
    """
    T_mat        - the transformation (rotation) matrix
    points_plane - the points on the plane
    """
    rotated  = np.transpose(T_mat @ np.transpose(points_plane)) # Apply rotation T to align screen to x-axis
    scrn     = rotated[:,1:]                                    # Kill off x dimension
    return scrn

# Get positions of unfilled pixels on the screen
def get_not_filled_positions(not_filled, n_x, n_y):
    count_not_filled     = np.sum(not_filled)
    not_filled_pos       = np.zeros((count_not_filled, 2))  # list of unfilled pixels by position
    
    not_filled_counter   = 0                                # counter for not filled
    for it_x in range(n_x):
        for it_y in range(n_y):
            if not_filled[it_x][it_y] == 1:
                not_filled_pos[not_filled_counter] = np.array([it_x, it_y])
                not_filled_counter += 1
    return not_filled_pos

# Get the direction in physical space of vectors from camera to missing pixels (this should be working)
def get_directional_vectors(not_filled_pos, n_x, n_y, a_x, a_y, b_x, b_y, A_mat, T_mat, cam):
    """
    not_filled - list of unfilled pixels
    All other parameters are standard as per its name
    
    Returns a list of unit vectors in physical space, which, when originating from cam, would pass through an unfilled pixel
    """         
    not_filled_counter   = not_filled_pos.shape[0]            # number of not filled pixels
    
    cam_in_flat          = T_mat @ cam                        # location of camera in the rotated space
    not_filled_x_pos     = not_filled_pos[:, 0]               # x-coordinate in the screen (this corresponds to y-space)
    not_filled_y_pos     = not_filled_pos[:, 1]               # y-coordinate in the screen (this corresponds to z-space)
    
    # (approximate) real coordinates of pixels in rotated space
    unshifted_x          = not_filled_x_pos*(b_x - a_x)/n_x + a_x + (b_x - a_x)/(2*n_x) # this last term is to re-center
    unshifted_y          = not_filled_y_pos*(b_y - a_y)/n_y + a_y + (b_y - a_y)/(2*n_y)
    unshifted_coords     = np.array([np.zeros(not_filled_counter), unshifted_x, unshifted_y])
    
    cam_to_pixels_in_rot = unshifted_coords - np.repeat(cam_in_flat.reshape(3,1), not_filled_counter, axis=1) # vectors originating from camera to pixels
    ctp_in_rot_normal    = cam_to_pixels_in_rot/np.linalg.norm(cam_to_pixels_in_rot, axis=0)                  # normalize
    ctp_in_space         = A_mat @ ctp_in_rot_normal                                                          # return to physical space - these are the directions for sphere tracing
    return ctp_in_space

# Perform sphere tracing given a vector
def perform_sphere_tracing(cam, direction, phi, dist_threshold = 10e-5, t_max = 5):
    t = 0                                   # distance along the ray
    try:
        while(t < t_max):
            radius = phi(direction*t + cam)  # radius at current point
            if radius < dist_threshold:
                break
            t += radius
        if t < t_max:
            return t
        else:
            return np.inf
    except:
        return np.inf # this will happen if the evaluation of the interpolant phi is outside the interpolation range