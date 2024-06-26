

"""
Taking parameters:
n_pixels: pixel dimensions of detector assuming square shape (flat 1d here)
radius: radius of assumed circular object to scan
phi: the angle at the source between the middle of the detector and the edge

Function determines and plots the distances:
d_source_obj: distance from source to object
d_obj_det: distance from object to detector
mag: Magnification of the object on the detector when doing the scan

"""

def lab_setup_2d_fanflat(n_pixels, radius, phi):
    import numpy as np
    
    d_source_obj = radius/np.tan(phi)
    ratio_d = radius/(n_pixels)
    d_obj_det = d_source_obj*ratio_d
    
    # Magnification of obj on detector due to fan beam
    mag = (d_source_obj + d_obj_det)/d_source_obj

    return d_source_obj, d_obj_det, mag


def plot_lab_setup_2d_fanflat(n_pixels, d_source_obj, d_obj_det, radius):
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_spacing = 50
    x1 = np.linspace(0, d_source_obj, n_spacing)
    y1 = np.ones(n_spacing)

    x2 = np.linspace(d_source_obj, d_source_obj + d_obj_det, n_spacing)
    y2 = y1
    
    x3 = np.ones(n_spacing)*(d_source_obj + d_obj_det)
    y3 = np.linspace(y1[0], y1[0] + n_pixels/2)
    
    x4 = np.linspace(x1[0], x1[0] + d_source_obj + d_obj_det, n_spacing)
    y4 = np.linspace(y1[0], y1[0] + n_pixels/2, n_spacing)
    
    x5 = np.ones(n_spacing)*(x1[0] + d_source_obj)
    y5 = np.linspace(y1[0], y1[0] + radius, n_spacing)
    
    # Plotting
    plt.figure()
    
    # Source to object
    plt.plot(x1,y1)
    
    # Object to detector
    plt.plot(x2,y2)
    
    # Detector going up
    plt.plot(x3,y3)
    
    # Source to edge of detector top
    plt.plot(x4,y4)
    
    # Radius of circular object
    plt.plot(x5,y5)
    
    plt.title("Lab Setup with angle phi of fan beam")
    plt.show()


if __name__ == "__main__":
    import numpy as np
    n_pixels = 200
    radius = 50
    
    d1, d2, mag = lab_setup_2d_fanflat(n_pixels, radius, np.pi*25/180)
    plot_lab_setup_2d_fanflat(n_pixels, d1, d2, radius)
    