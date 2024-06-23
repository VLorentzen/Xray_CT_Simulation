# Standard packages
import numpy as np
import matplotlib.pyplot as plt

# Packages specifically for X-ray absorption spectroscopy
import astra
import spekpy

# Local/custom functions
from create_obj import default_obj_cfg, create_obj
from astra_functions import create_projector, polychromatic_sinogram, polychromatic_sinogram_multiple



## X-ray source definitions yielding a polychromatic beam and a beam hardened sinogram
material = 'C'
target_material = 'W'
filter_material = 'Al'
beam = 'parallel3d'
#beam = 'fanflat'
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
spek_spectrum= spekpy.Spek(kvp = acceleration_voltage/1000, th  = anode_angle) # Create a spectrum, energy in keV
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
n_pixels_x = n_pixels
n_pixels_y = n_pixels
n_pixels = [n_pixels_x, n_pixels_y]

dim = 2
n_angles = round(max(n_pixels) *np.pi/2)                    # Theoretical value for needed angles to create good reconstruction
proj_angles = np.linspace(-np.pi, np.pi, 2*n_angles)  # Angles of projection



## FROM
#
# d_source_obj + d_obj_det = n_pixels
# magnification = (d_source_obj + d_obj_det) / d_source_obj
#
# =>

magnification = 2
d_source_obj = max(n_pixels)/magnification
d_obj_det = max(n_pixels) - d_source_obj



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
poly_sinogram_array = []

rec_id_array = []
rec_array = []
rec_harden_id_array = []
rec_harden_array = []

# create multiple objs with different attenuation
for i in range(len(obj_dim_array)):
    
    # Projection and sinogram
    proj_id = create_projector(obj_array[i], d_source_obj, d_obj_det, n_pixels, proj_angles, beam)
    sinogram_id, sinogram = astra.create_sino(obj_array[i], proj_id)
    sinogram = np.transpose(sinogram)
    poly_sinogram = polychromatic_sinogram(sinogram, material_array[i], energy_spectrum, intensity_spectrum)
    print("converted sino to poly")
    
    proj_id_array.append(proj_id)
    sinogram_id_array.append(sinogram_id)
    sinogram_array.append(sinogram)
    poly_sinogram_array.append(poly_sinogram)
    
    plt.figure()
    if (dim == 2):
        plt.imshow(obj_array[i])
    elif (dim == 3):
        
        # Just a slice of the 3d obj
        plt.imshow(obj_array[i][:,:,round(np.shape(obj)[2]/2)])
    plt.colorbar()
    plt.title("Original Shape")
    plt.show()

    plt.figure()
    plt.imshow(sinogram_array[i])
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(poly_sinogram_array[i])
    plt.colorbar()
    plt.show()
    
    
    ## 2D Reconstruction
    [rec_id, rec] = astra.creators.create_reconstruction("FBP", proj_id_array[i], np.transpose(sinogram_array[i]))
    rec_id_array.append(rec_id)
    rec_array.append(rec)

    plt.figure()
    plt.imshow(rec_array[i])
    plt.colorbar()
    plt.title("Reconstruction of obj " + str(i + 1))
    plt.show()
    
    plt.figure() 
    plt.plot( rec_array[i][ round( centers_array[i][0][0]),: ])
    plt.title("Reconstruction profile of obj " + str(i + 1))
    plt.show()


    ## 2D Reconstruction of Beam Hardened
    [rec_harden_id, rec_harden] = astra.creators.create_reconstruction("FBP", proj_id_array[i], np.transpose(poly_sinogram_array[i]) )
    rec_harden_id_array.append(rec_harden_id)
    rec_harden_array.append(rec_harden)
    
    plt.figure()
    plt.imshow(rec_harden_array[i])
    plt.colorbar()
    plt.title("Beam Hardened Reconstruction of obj " + str(i + 1))
    plt.show()

    plt.figure() 
    plt.plot( rec_harden_array[i][ round( centers_array[i][0][0] ),: ])
    plt.title("Beam Hardened Reconstruction profile of obj " + str(i + 1))
    plt.show()

    
    

# Projection and sinogram FOR JUST A SINGLE OBJ MASKING
#proj_id = create_projector(obj, d_source_obj, d_obj_det, n_pixels_x, n_pixels_y, dim, proj_angles, beam)
#sinogram_id, sinogram = astra.create_sino(obj, proj_id)
#sinogram = np.transpose(sinogram)
#poly_sinogram = polychromatic_sinogram(sinogram, material, energy_spectrum, intensity_spectrum)
#print("converted sino to poly")


## Now to add them all together
obj_combined = np.sum(obj_array, axis=0)
sinogram_combined = np.sum(sinogram_array, axis=0)
#poly_sinogram_combined = np.sum(poly_sinogram_array, axis=0)
poly_sinogram_combined = polychromatic_sinogram_multiple(sinogram_array, material_array, energy_spectrum, intensity_spectrum)
rec_combined = np.sum(rec_array, axis=0)
rec_harden_combined = np.sum(rec_harden_array, axis=0)


plt.figure()
plt.imshow(sinogram_combined)
plt.title("Combined sinogram")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(poly_sinogram_combined)
plt.title("Combined polychromatic sinogram")
plt.colorbar()
plt.show()

# Testing if the polychromatic is done right...
plt.figure()
plt.imshow(sinogram_combined - poly_sinogram_combined*(np.sum(sinogram_combined)/np.sum(poly_sinogram_combined)))
plt.title("Diff between scaled sinogram and polychromatic sinogram")
plt.colorbar()
plt.show()


[rec_id_final, rec_final] = astra.creators.create_reconstruction("FBP", proj_id, np.transpose(sinogram_combined))
plt.figure()
plt.imshow(rec_final)
plt.colorbar()
plt.title("Reconstruction of combined sinogram")
plt.show()

plt.figure() 
plt.plot( rec_final[ round( np.size( rec_final,0 )/2 ),: ])
plt.title("Reconstruction profile of combined sinogram")
plt.show()


## 2D Reconstruction of Beam Hardened
# Transposing the sinogram back to astras form
[rec_harden_id_final, rec_harden_final] = astra.creators.create_reconstruction("FBP", proj_id, np.transpose(poly_sinogram_combined) )

plt.figure()
plt.imshow(rec_harden_final)
plt.colorbar()
plt.title("BH Reconstruction of combined sinogram")
plt.show()

plt.figure() 
plt.plot( rec_harden_final[ round( np.size( rec_harden_final,0 )/2 ),: ])
plt.title("BH Reconstruction profile on combined sinogram")
plt.show()



plt.figure()
plt.imshow(rec_combined)
plt.colorbar()
plt.title("Combined Reconstruction")
plt.show()

plt.figure() 
plt.plot( rec_combined[ round( np.size( rec_combined,0 )/2 ),: ])
plt.title("Combined Reconstruction profile")
plt.show()



plt.figure()
plt.imshow(rec_harden_combined)
plt.colorbar()
plt.title("Combined BH Reconstruction")
plt.show()

plt.figure() 
plt.plot( rec_harden_combined[ round( np.size( rec_harden_combined,0 )/2 ),: ])
plt.title("Combined BH Reconstruction profile")
plt.show()



plt.figure()
plt.imshow(rec_combined - rec_final)
plt.title("Linearity of reconstruction")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(rec_harden_combined - rec_harden_final)
plt.title("Linearity of reconstruction with BH")
plt.colorbar()
plt.show()



plt.figure()
plt.imshow(rec_harden_combined - rec_combined)
plt.title("Reconstruction BH - reconstruction added")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(rec_harden_final - rec_final)
plt.title("Reconstruction BH - reconstruction total")
plt.colorbar()
plt.show()



plt.figure()
plt.imshow(rec_combined - obj_combined)
plt.title("Reconstruction vs true obj")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(rec_harden_combined - obj_combined)
plt.title("Reconstruction BH vs true obj")
plt.colorbar()
plt.show()