# Standard packages
import numpy as np
import matplotlib.pyplot as plt

# Packages specifically for X-ray absorption spectroscopy
import astra
import spekpy

# Local/custom functions
from modules.create_obj import default_obj_cfg, create_obj
from modules.astra_functions import create_projector, polychromatic_sinogram, polychromatic_sinogram_multiple



## X-ray source definitions yielding a polychromatic beam and a beam hardened sinogram
material = 'C'
target_material = 'W'
filter_material = 'Al'
beam = 'parallel'


# Change filter thickness to 0 no matter which filter and you cancel it
filter_thickness = 4E-3 #m
anode_angle = 12 #deg
acceleration_voltage = 80E3 #eV

# Other settings used in the lab
current = 667E-6 #A
exposure_time = 500E-3 #s
voxel_size = 132E-6 #m
n_angles_exp = 1570 #or 360


## SpekPy to generate the X-ray source Spectrum
spek_spectrum = spekpy.Spek(kvp = acceleration_voltage/1000, th  = anode_angle) # Create a spectrum, energy in keV
spek_spectrum.filter(filter_material, filter_thickness*1000) # Filter the spectrum, thickness in mm?
energy_spectrum, intensity_spectrum = spek_spectrum.get_spectrum(edges=True) # Get the spectrum



plt.plot(energy_spectrum, intensity_spectrum) # Plot the spectrum
plt.xlabel('Energy [keV]')
plt.ylabel('Fluence per mAs per unit energy [photons/cm2/mAs/keV]')
plt.title('X-ray source spectrum')
plt.show()



## Lab setup including detection volume, detector dimensions, sample dimensions
# Square detector
n_pixels = 200
dim = 2

n_pixels_x = n_pixels
n_pixels_y = n_pixels
if (dim == 2):
    n_pixels = n_pixels_x
    max_n_pixels = n_pixels_x
else:
    n_pixels = [n_pixels_x, n_pixels_y]
    max_n_pixels = max(n_pixels)

n_angles = round(max_n_pixels*np.pi/2)                    # Theoretical value for needed angles to create good reconstruction
proj_angles = np.linspace(-np.pi, np.pi, 2*n_angles)        # Angles of projection



## FROM
#
# d_source_obj + d_obj_det = n_pixels
# magnification = (d_source_obj + d_obj_det) / d_source_obj
#
# =>

magnification = 2
d_source_obj = max_n_pixels/magnification
d_obj_detector = max_n_pixels - d_source_obj



# Getting a circular object to test on
obj_dim = [n_pixels_x, n_pixels_y]
centers = [[120,145]]
shapes = ['circle']
radii = [40]

#obj = create_obj(obj_dim, centers, shapes, radii)

obj_dim_array = [obj_dim]
centers_array = [centers]
shapes_array = [shapes]
radii_array = [radii]
material_array = [material]


obj_dim = [n_pixels_x, n_pixels_y]
centers = [[120,55]]
shapes = ['circle']
radii = [40]
material = 'Cu'

obj_dim_array.append(obj_dim)
centers_array.append(centers)
shapes_array.append(shapes)
radii_array.append(radii)
material_array.append(material)


obj_dim = [n_pixels_x, n_pixels_y]
centers = [[50,100]]
shapes = ['circle']
radii = [40]
material = 'Pb'

obj_dim_array.append(obj_dim)
centers_array.append(centers)
shapes_array.append(shapes)
radii_array.append(radii)
material_array.append(material)



obj_array = []
for i in range(len(obj_dim_array)):
    obj = create_obj(obj_dim_array[i], centers_array[i], shapes_array[i], radii_array[i])
    obj = obj*(i + 1)
    obj_array.append(obj)

# Removing overlap from the objects
obj_total = np.sum(obj_array, axis=0)
#obj_total[obj_total > 1] = 1
#for i in range(len(obj_dim_array) - 1):
#    obj_array[i + 1] = obj_total - obj_array[i]
#    plt.figure()
#    plt.imshow(obj_array[i+1])
#    plt.show()



proj_id_array = []
sinogram_id_array = []
sinogram_array = []

rec_id_array = []
rec_array = []
rec_harden_id_array = []
rec_harden_array = []



# create multiple objs with different attenuation
for i in range(len(obj_dim_array)):
    
    # Projection and sinogram
    proj_id = create_projector(obj_array[i], d_source_obj, d_obj_detector, n_pixels, proj_angles, beam)
    sinogram_id, sinogram = astra.create_sino(obj_array[i], proj_id)
    sinogram = np.transpose(sinogram)

    proj_id_array.append(proj_id)
    sinogram_id_array.append(sinogram_id)
    sinogram_array.append(sinogram)

    

## Now to add them all together
obj_combined = np.sum(obj_array, axis=0)
poly_sinogram_combined = polychromatic_sinogram_multiple(sinogram_array, material_array, energy_spectrum, intensity_spectrum)
rec_harden_combined = np.sum(rec_harden_array, axis=0)


plt.figure()
plt.imshow(poly_sinogram_combined)
plt.title("Polychromatic Beam Sinogram")
plt.colorbar()
plt.show()


## 2D Reconstruction of Beam Hardened
# Transposing the sinogram back to astras form
[rec_harden_id_final, rec_harden_final] = astra.creators.create_reconstruction("FBP", proj_id, np.transpose(poly_sinogram_combined) )

plt.figure()
plt.imshow(rec_harden_final)
plt.colorbar()
plt.title("BH Specimen Reconstruction")
plt.show()

plt.figure() 
plt.plot( rec_harden_final[ round( np.size( rec_harden_final,0 )/2 ),: ])
plt.title("BH Specimen Reconstruction: profile")
plt.show()

plt.figure()
plt.imshow(rec_harden_combined)
plt.colorbar()
plt.title("Combining BH Specimen after Reconstruction")
plt.show()

plt.figure() 
plt.plot( rec_harden_combined[ round( np.size( rec_harden_combined,0 )/2 ),: ])
plt.title("Combining BH Specimen after Reconstruction: profile")
plt.show()

plt.figure()
plt.imshow(rec_harden_combined - rec_harden_final)
plt.title("Beam Hardening Artifacts")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(rec_harden_combined - obj_combined)
plt.title("BH Specimen Reconstruction vs True Specimen")
plt.colorbar()
plt.show()