# Import Libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn import preprocessing

# Import functions
from recreate_sdf import recreate
from stochastic_raytracer_helper import *

""" PARAMETERS - EDIT ONLY THESE """

space_subdivs = 200                  # Number of subdivisions on S

recreate_sdf  = False                # Recreate the SDF
dict_name     = 'sdf_cylinder_z.dictionary'

cam           = np.array([-1,-1,-1]) # Location of the camera
A, B, C, D    = 1, 1, 1, 2           # Screen given by the plane Ax + By + Cz + D = 0, this should be perpendicular to the vector given by cam

light         = np.array([-1,-1,-1])    # Location of light source
intensity     = 200                  # intensity of the light (integer between 0 and 255)
ambient       = 10                   # ambient intensity of the light onto the object applied uniformly

a_x, b_x      = -.1, .1              # Extent of screen in x coordinates
a_y, b_y      = -.1, .1              # Extent of screen in y coordinates
density       = 1000                 # Number of pixels per unit of screen

batches       = 1                    # Number of batches
N             = 3000000              # Points to generate at each batch

outside_sdf   = False                # Only use points with positive SDF values - useful for union/intersection of simple SDFs - incompatible with batches

ave_method    = 'x'                  # Method for taking average to patch holes, if 'x' then average of generated points, if 'y' then average of snapped points

fix_seed      = 0                    # The random seed, set to 0 if not using seed

""" END PARAMETERS """


""" STEP 0 - PRELIMINARY SET UP """

if fix_seed > 0:
    np.random.seed(fix_seed)

# The number of collisions in the image (two or more rays going through the same pixel)
collisions = 0 


""" The space S, in this case a cube of side length 2 centered at the origin """

x   = np.linspace(-1, 1, space_subdivs)
y   = np.linspace(-1, 1, space_subdivs)
z   = np.linspace(-1, 1, space_subdivs)
xyz = (x,y,z)


""" Signed Distance Function """

# If necessary, recreate SDF and save as a dict file
if recreate_sdf:
    recreate()
        
# Get SDF and gradient
phi, fx, fy, fz         = get_sdf_components(dict_name, xyz, space_subdivs)

# Bounding sphere SDF (this is a sphere of radius 1 centered at the origin)
phi_b, fx_b, fy_b, fz_b = get_sdf_components('sdf_boundingsphere.dictionary', xyz, space_subdivs)


""" Pixels """

n_x           = int((b_x - a_x)*density)            # number of pixels in x
n_y           = int((b_y - a_y)*density)            # number of pixels in y

pixels        = np.zeros((n_x, n_y), dtype='int32') # initialize all pixels to 0
pixels_b      = np.zeros((n_x, n_y), dtype='int32')
pixels_dist   = 10000*np.ones((n_x, n_y))           # initialize all distances to 10000 - arbitrary very large number
pixels_dist_b = 10000*np.ones((n_x, n_y))
pixels_corr   = np.zeros((n_x, n_y), dtype='int32') # this will be populated with the index of the point whose closest point was traced back

cp_list       = np.zeros((batches*N, 3))            # this keeps track of all the closest points that were generated
cp_list_b     = np.zeros((batches*N, 3))


""" Rotation matrix """

# Vector normal to the plane, in the direction towards shape
k = np.array((A,B,C))
n = k/np.linalg.norm(k) # Normalize normal vector

A_mat, T_mat  = get_rotation_matrix(k, cam)


""" STEP 1 - GENERATE RANDOM POINTS AND PROJECT """

#Run the routine batches-number of times (batches is currently unused)
for q in range(batches):
    print("Generating batch {}".format(q))
    
    """ Generation of points and computation of closed point on the SDF zero level-set """    
    
    # Generate points in [-1, 1]^3
    p                      = 2*np.random.rand(N, 3)-1
    
    # Delete inside points if applicable
    if outside_sdf:
        p = p[phi(p) > 0]
        N = len(p)
    
    # Compute closest point and store info
    closest                = cp(p, phi, fx, fy, fz)
    #cp_list                = closest
    cp_list[q*N:(q+1)*N]   = closest # HOTFIX: disable batches for outside_sdf
    
    closest_b              = cp(p, phi_b, fx_b, fy_b, fz_b)
    #cp_list_b              = closest_b
    cp_list_b[q*N:(q+1)*N] = closest_b
    
    
    """ Compute the intersected points on the screen"""

    points_plane, distances     = get_points_on_plane(cam, closest, k, n, D)
    points_plane_b, distances_b = get_points_on_plane(cam, closest_b, k, n, D)
    
    
    """ Rotate to remove one dimension of the screen """
    
    scrn   = rotate_and_kill_x(T_mat, points_plane)
    scrn_b = rotate_and_kill_x(T_mat, points_plane_b)
    
    
    """ Screen """
    
    # Screen borders - this is done by removing one dimension (x) and partitioning the plane R^2 (in terms of new x, y)
    # Border is [a_x, b_x] in each of x (and similarly in y), partitioned into based on pixel density defined in parameters
    
    # Shift over to the rectangle [0, b_x - a_x] x [0, b_y - a_y]
    
    loc           = scrn - np.array([a_x, a_y]).reshape(1,2) # set the bottom left to be the 'zeros' pixel
    loc_x         = loc[:,0]/(b_x - a_x)                     # determine location on screen of x
    loc_y         = loc[:,1]/(b_y - a_y)                     # determine location on screen of y
    
    pixel_num_x   = np.floor(loc_x*n_x)                      # determine corresponding pixel in x
    pixel_num_y   = np.floor(loc_y*n_y)                      # determine corresponding pixel in y
    
    # Populate the pixels based on the closest ray that goes through partition
    for it in range(N):
        # coordinates of the it-th pixel as (i,j)
        i = int(pixel_num_x[it])
        j = int(pixel_num_y[it])
        
        # disregard coordinates that fall outside of the screen
        if i < 0 or j < 0 or i >= n_x or j >= n_y:
            continue
        
        if pixels_dist[i][j] < 5000:          # 5000 is arbitrary large number since the distance is initiated to 10000
            collisions += 1                   # counting the number of collisions in a pixel
        
        if distances[it] < pixels_dist[i][j]:
            pixels_dist[i][j] = distances[it] # update closest distance
            pixels[i][j]      = 1             # turn pixel on
            pixels_corr[i][j] = N*q + it      # remember the index of the point that populates this pixel
    
    # Repeat procedure for bounding sphere
    loc_b         = scrn_b - np.array([a_x, a_y]).reshape(1,2)
    loc_x_b       = loc_b[:,0]/(b_x - a_x)
    loc_y_b       = loc_b[:,1]/(b_y - a_y)
    
    pixel_num_x_b = np.floor(loc_x_b*n_x)
    pixel_num_y_b = np.floor(loc_y_b*n_y)
    
    for it in range(N):
        # coordinates of the it-th pixel as (i,j)
        i = int(pixel_num_x_b[it])
        j = int(pixel_num_y_b[it])
        
        # disregard coordinates that fall outside of the screen
        if i < 0 or j < 0 or i >= n_x or j >= n_y:
            continue
        
        # collision check unnecessary
        
        if distances_b[it] < pixels_dist_b[i][j]:
            pixels_dist_b[i][j] = distances_b[it]
            pixels_b[i][j]      = 1
            
            
""" STEP 2 - ATTEMPT TO PATCH HOLES """
# Note: Currently only supports batches = 1

min_neighbours = 2                                # Minimum number of neighbours to compute average
holes          = np.zeros((n_x, n_y, 3))          # Storage of average of generated points / closest points
#holes_loc      = np.zeros((n_x, n_y, 2))          # Storage of index

p_use = (p if ave_method == 'x' else closest)

# Ignore edge pixels
for it_x in range(1, n_x-1):
    for it_y in range(1, n_y-1):
        if pixels[it_x][it_y] == 0:
            neighbour_count = pixels[it_x-1][it_y] + pixels[it_x+1][it_y] + pixels[it_x][it_y-1] + pixels[it_x][it_y+1] # number of filled adjacent cells
            if neighbour_count >= min_neighbours:
                north = pixels_corr[it_x][it_y+1] # Get index of neighbouring pixels
                south = pixels_corr[it_x][it_y-1]
                east  = pixels_corr[it_x+1][it_y]
                west  = pixels_corr[it_x-1][it_y]
                avg_point = np.zeros(3)           # This will store the average point that generates neighbours
                if north != 0:                    # Add non-zero entries to the average point
                    avg_point += p_use[north]
                if south != 0:
                    avg_point += p_use[south]
                if east != 0:
                    avg_point += p_use[east]
                if west != 0:
                    avg_point += p_use[west]
                holes[it_x][it_y]     = avg_point/neighbour_count
                #holes_loc[it_x][it_y] = [it_x, it_y]

n_holes  = np.sum(np.where(holes[:,:,0] != 0, 1, 0)) # Number of eligible holes
holes_a  = holes[:,:,0].flatten()                    # Flattening in each axis
holes_b  = holes[:,:,1].flatten()
holes_c  = holes[:,:,2].flatten()

#h_loc_a  = holes_loc[:,:,0].flatten()                # Flattening location in each axis
#h_loc_b  = holes_loc[:,:,1].flatten()

fillable = np.array([holes_a[holes_a != 0],holes_b[holes_b != 0],holes_c[holes_c != 0]]).T # The (n_holes, 3) list of points
#fill_loc = np.array([h_loc_a[h_loc_a != 0],h_loc_b[h_loc_b != 0]]).T

# Apply above routine to these extra points
fillable_closest            = cp(fillable, phi, fx, fy, fz)
cp_list                     = np.append(cp_list, fillable_closest, axis=0)
fill_pts_plane, distances_f = get_points_on_plane(cam, fillable_closest, k, n, D)
fillable_scrn               = rotate_and_kill_x(T_mat, fill_pts_plane)

loc_f         = fillable_scrn - np.array([a_x, a_y]).reshape(1,2)
loc_x_f       = loc_f[:,0]/(b_x - a_x)
loc_y_f       = loc_f[:,1]/(b_y - a_y)

pixel_num_x_f = np.floor(loc_x_f*n_x)
pixel_num_y_f = np.floor(loc_y_f*n_y)

filled_count  = 0 # Tracks number of fillable holes that are actually filled

for it in range(n_holes):
    # coordinates of the it-th pixel as (i,j)
    i = int(pixel_num_x_f[it])
    j = int(pixel_num_y_f[it])
    
    # disregard coordinates that fall outside of the screen
    if i < 0 or j < 0 or i >= n_x or j >= n_y:
        continue
    
    # use the same distances as in the original iteration
    if distances_f[it] < pixels_dist[i][j]:
        pixels_dist[i][j] = distances_f[it]
        pixels[i][j]      = 1
        pixels_corr[i][j] = N + it # this will be added to the end of CP list
        filled_count     += 1
        
""" STEP 3 - APPLY STANDARD SPHERE TRACING """
# April 24: This currently doesn't do anything but produce a few plots
# the plots suggest the inverse mapping from rotated space to physical space is incorrect
# additionally, not_filled is much less populated than expected

if False:
    not_filled          = 1 - np.maximum(pixels, pixels_b)               # pixels that have yet to be filled by the object or the background as 2-d array
    not_filled          = 1 - pixels #tempfix
    not_filled_pos      = get_not_filled_positions(not_filled, n_x, n_y) # list of positions of pixels that are unfilled
    directional_vectors = get_directional_vectors(not_filled_pos, n_x, n_y, a_x, a_y, b_x, b_y, A_mat, T_mat, cam)
    
    nb_unfilled         = not_filled_pos.shape[0]                        # number of unfilled pixels
    sphere_trace_dists  = np.zeros(nb_unfilled)                          # storage for distances of each sphere tracing
    traced_points       = np.zeros((nb_unfilled, 3))                     # the point on the shape obtained from sphere tracing
    
    for it_sp in range(nb_unfilled):                                     # perform the sphere tracing for every direction
        sphere_trace_dists[it_sp] = perform_sphere_tracing(cam, directional_vectors[:,it_sp], phi)
        traced_points[it_sp]      = cam + sphere_trace_dists[it_sp]*directional_vectors[:,it_sp]
    
    # temp plots - currently indicating that inverse mapping is incorrect
    plt.scatter(traced_points.T[0],traced_points.T[1], s=1)
    plt.show()
    plt.scatter(traced_points.T[0],traced_points.T[2], s=1)
    plt.show()
    plt.scatter(traced_points.T[1],traced_points.T[2], s=1)
    plt.show()


""" STEP 4 - APPLY LAMBERTIAN MODEL """

pixels_corr_flat  = pixels_corr.flatten()             # index of corresponding pixels flattened into a (n_x * n_y, 1) vector
cp_corr_flat      = cp_list[pixels_corr_flat]         # corresponding pixels as a (n_x * n_y, 3) vector
L                 = light.reshape(1,3) - cp_corr_flat # vectors from closest point to the light source
L                 = L/np.linalg.norm(L, axis=1).reshape(-1,1)
corr_flat_grad    = np.array((fx(cp_corr_flat), fy(cp_corr_flat), fz(cp_corr_flat))) # gradient at closest points

pixel_intensity   = np.zeros((n_x, n_y))              # Initialize intensity to 0
pixel_intensity   = L * np.transpose(corr_flat_grad)  # pixel intensity as a (n_x * n_y, 3) vector
pixel_intensity   = pixel_intensity.sum(axis=1)       # sum to get the final intensity
pixel_intensity   = pixel_intensity.reshape(n_x, n_y) # reshape
pixel_intensity  *= pixels                            # keep only those that are on (others might be erroneously on due to index 0)
pixel_intensity   = np.maximum(pixel_intensity, 0)    # keep only positive values
pixel_intensity  *= intensity                         # scale up by the intensity
pixel_intensity   = np.minimum(pixel_intensity, 255)  # cap pixel intensity at 255

""" Display screen """

output         = np.repeat(np.floor(pixel_intensity), 3).astype('int32').reshape(200, 200, -1) # Pixel intensity as RGB
#output[:,:,2] += 80*pixels_b*(1-pixels)                                                        # Add blue to back-sphere pixels (removing pixels which connect to surface of object)
output        += np.repeat(ambient*pixels, 3).astype('int32').reshape(200, 200, -1)            # Add ambient lighting to the surface
#output[:,:,0]  = 0                                                                             # Remove all blue light

plt.figure(figsize=(80*(b_x-a_x),80*(b_y-a_y)))
plt.imshow(output, origin='lower')
