# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:58:39 2024

@author: Victor
"""


def default_obj_cfg(n_pixels, dim):

    import numpy as np
    obj_dim = np.ones(dim, dtype = int)
    obj_dim = obj_dim*n_pixels
    
    #radii = [round(n_pixels/4)]
    radii = [n_pixels/4]
    centers = np.array([np.ones(dim, dtype = int)])
    #centers = centers*round(n_pixels/2)
    centers = centers*(n_pixels - 1)/2
    #centers = centers*0
    shapes = ['circle']
    
    return obj_dim, centers, shapes, radii


def create_obj(obj_dim, centers, shapes, radii):
    """
    All inputs are arrays with elements determining each object in the sample 

    Inputs
    obj_dim:    Dimensions of the object image, [X,Y] or [X,Y,Z]
    centers:    Coordinates of [X,Y] for center of objects, [X,Y,Z] for 3D
    shapes:     'circle', 'square', ... 
    radii:      radius for circles, shortest distance from center to edge for squares
    
    Outputs
    obj:        Object created from inputs, np.array of 2d or 3d obj
    """
    import numpy as np
    
    # Is object 2D or 3D?
    if (len(obj_dim) == 2):
        
        # Meshgrid using X,Y and corresponding object dimensions, obj_dim
        [X, Y] = np.meshgrid(np.linspace(0, obj_dim[0] - 1, obj_dim[0]), np.linspace(0, obj_dim[1] - 1, obj_dim[1]))
        
        # Predefine object as zeros
        obj = np.zeros( ( np.size(X,0),np.size(Y,1) ), dtype = float)
    
        for i in range(len(centers)):
            
            # Check the shape of the current object
            if (shapes[i] == 'circle'):
                
                # Equation for a circle
                obj += (X-centers[i][0])**2 + (Y-centers[i][1])**2 < radii[i]**2
            elif (shapes[i] == 'square'):
                
                # Equation for a square ... More correct X^inf + Y^inf = radius^inf
                obj += (X-centers[i][0])**8 + (Y-centers[i][1])**8 < radii[i]**8
                
    elif (len(obj_dim) == 3):
        
        # Meshgrid using X,Y,Z and corresponding object dimensions, obj_dim
        [X, Y, Z] = np.meshgrid(np.linspace(0, obj_dim[0] - 1, obj_dim[0]), np.linspace(0, obj_dim[1] - 1, obj_dim[1]), np.linspace(0, obj_dim[2] - 1, obj_dim[2]))
        
        # Predefine object as zeros
        obj = np.zeros( ( np.size(X,0),np.size(Y,1), np.size(Z,2) ), dtype = float)
        
        for i in range(len(centers)):
            
            # Check the shape of the current object
            if (shapes[i] == 'circle'):
                
                # Equation for a circle
                obj += (X-centers[i][0])**2 + (Y-centers[i][1])**2 + (Z-centers[i][2])**2 < radii[i]**2
            elif (shapes[i] == 'square'):
                
                # Equation for a square ... More correct X^inf + Y^inf + Z^inf = radius^inf
                obj += (X-centers[i][0])**8 + (Y-centers[i][1])**8 + (Z-centers[i][2])**8 < radii[i]**8
    
    # Object must be 2D or 3D - 1D gives 0D detector.
    # 0D detector and 4D object are unphysical, so must be 2D or 3D
    else:
        print("Object must have 2 or 3 dimensions")

    obj[obj > 1] = 1
    return obj

if __name__ == "__main__":
    n_pixels = 200
    dim = 3

    import numpy as np
    import matplotlib.pyplot as plt

    obj_dim, centers, shapes, radii = default_obj_cfg(n_pixels, dim)
    obj = create_obj(obj_dim, centers, shapes, radii)
    
    # Other syntax that is allowed but doesn't save variables
    #obj = create_obj( *default_obj_cfg(n_pixels, dim) )
    
    if dim == 3:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage import measure
        verts, faces, normals, values = measure.marching_cubes(obj, 0)

        # Display resulting triangular mesh using Matplotlib. This can also be done
        # with mayavi (see skimage.measure.marching_cubes docstring).
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)

        ax.set_xlabel("x-axis: a = 6 per ellipsoid")
        ax.set_ylabel("y-axis: b = 10")
        ax.set_zlabel("z-axis: c = 16")

        ax.set_xlim(0, 200)  # a = 6 (times two for 2nd ellipsoid)
        ax.set_ylim(0, 200)  # b = 10
        ax.set_zlim(0, 200)  # c = 16

        plt.tight_layout()
        plt.show()
        
        plt.figure()
        plt.imshow(obj[round(np.shape(obj)[0]/2),:,:])
        plt.title("Slice in X-axis middle 2D")
        plt.show()
        
        plt.figure()
        plt.imshow(obj[round(np.shape(obj)[0]*2/3),:,:])
        plt.title("Slice in X-axis 66% through 2D")
        plt.show()
        
    elif dim == 2:
        plt.figure()
        plt.imshow(obj)
        plt.colorbar()
        plt.title("Testing object creation function 2d")
        plt.show()
        
    
    
    
    
    