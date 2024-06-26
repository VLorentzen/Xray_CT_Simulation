import numpy as np
import matplotlib.pyplot as plt
import os

# get the file path using file_name
file_name = '\heat_sink_sinogram.tif'
path = os.getcwd() + '\data\lab_experiment'

# read in sinogram as .tif image
sinogram = plt.imread(path + file_name)

# transpose to get from (pixel, angle) to (angle, pixel)
sinogram = np.transpose(sinogram)

# sinogram is transmission sinogram, so change to absorption
sinogram_max = np.max(sinogram) 
sinogram = sinogram_max - sinogram

# plot the results
plt.figure()
plt.imshow(sinogram)
plt.title("Imported Absorption Sinogram")
plt.xlabel("Angle Iteration")
plt.ylabel("Detector pixel")
plt.colorbar()
plt.show()



[rec_harden_id, rec_harden] = astra.creators.create_reconstruction("FBP", proj_id, np.transpose(sinogram))