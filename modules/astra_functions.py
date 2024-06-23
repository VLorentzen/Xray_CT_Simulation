"""
Goal
    The purpose of this script is to gather ASTRA functions that together 
    generate results in the following order:
    1. func: create_projector()
        1.1 object (non-ASTRA, custom to this script)
        1.2 Volume Geometries (ASTRA)
        1.3 Projection Geometries (ASTRA)
        1.4 Projector from beam type (ASTRA)
        1.5 Sinogram from Projection and object
        1.6 Reconstruction of object from Sinogram'
        
    2. func: polychromatic_sinogram (all custom to this script)
        2.1 Energy spectrum from source/detector settings (E) [keV]
        2.2 Definition of geometries in real dimensions (x,y,z)[m]
        2.3 Attenuation coefficients (um)[cm^2/g] convert to [m^2/kg] for given energies (E)[keV] 
               1 [cm^2/g] = 0.1 [m^2/kg] 
        2.4 Apply thickness (d)[m] from sinogram to Lambert Beer's Law 
            (absorption exponential function of attenuation and distance (d)[m])
        2.5 Reconstruction of new sinogram with polychromatic X-Ray source
            showing beam hardening
    
    3. Apply spot blurring by using projection_vec geometry
    4. Dead Pixel on detector
    5. Spread of reception on detector
    6. Detector pixel size limits?
    7. Other notes from Trello Board
    
Author: Victor Lorentzen
"""


def create_projector(obj, d_source_obj, d_obj_detector, n_pixels, proj_angles, beam, testing = False):
    try:
        dim = len(n_pixels)
    except:
        dim = 2
        n_pixels_x = n_pixels
        n_pixels_y = 0
    else:
        if (dim > 3):
            raise ValueError("Dimensions unphysical, must be 2D or 3D")
        n_pixels_x = n_pixels[0]
        n_pixels_y = n_pixels[1]


    import astra
    
    d_source_detector = int(round(d_source_obj + d_obj_detector))
    
    # Define volume geometry using ASTRA
    if (dim == 2):
        vol_geom = astra.create_vol_geom(n_pixels_x, d_source_detector)
        
    elif(dim == 3):
        vol_geom = astra.create_vol_geom(n_pixels_x, d_source_detector, n_pixels_y)

    
    ## DETECTOR DEFINITIONS
    # 2 dimensions are max, x,y. Basic step size used as 1.0
    
    # 1st dimension of detector in x direction
    detector_spacing_x = 1.0
    detector_row_count = n_pixels_x
    
    # 2nd dimension of detector in y direction
    detector_spacing_y = 1.0
    detector_column_count = n_pixels_y
    
    
    ## Create Projection through Astra ##
    """ 
    Docs:
        https://astra-toolbox.com/docs/proj2d.html
        
    
    Beam projector types:
        parallel:   2d parallel
        fanflat:    2d fan beam (fan beams always 2d)
        parallel3d: 3d parallel
        cone...
        
    Ray types:
        line:       0 thickness ray traced from source to detector
        strip:      pixel thickness ray from source to each detector
        linear:     thickness ray and linear interpolation (Joseph-kernel/slice-interpolated kernel)
    
    """
    
    # 2d Parallel projection initialization
    if ('parallel' == beam):
        proj_geom = astra.create_proj_geom('parallel', detector_spacing_x, detector_row_count, proj_angles)
        
        # Supported: line, strip, linear
        proj_id = astra.create_projector('strip', proj_geom, vol_geom)
    
    # 2d Fan(flat) beam projection initialization
    elif ('fanflat' == beam):    
        proj_geom = astra.create_proj_geom('fanflat', detector_spacing_x, detector_row_count, proj_angles, d_source_obj, d_obj_detector)
        
        # Supported: line_fanflat, strip_fanflat
        proj_id = astra.create_projector('strip_fanflat', proj_geom, vol_geom)
    
    # 3d Parallel projection initialization
    elif ('parallel3d' == beam):
        proj_geom = astra.create_proj_geom('parallel3d', 
                                           detector_spacing_x, 
                                           detector_spacing_y, 
                                           detector_row_count, 
                                           detector_column_count, 
                                           proj_angles, 
                                           d_source_obj, 
                                           d_obj_detector
                                           )
        
        # Currently not working, can stack 2d parallel instead
        proj_id = astra.create_sino3d_gpu(obj, proj_geom, vol_geom)
        
    # 3d Cone projection initialization. Cone is 3d version of fan
    elif ('cone' == beam):
        proj_geom = astra.create_proj_geom('cone', 
                                           detector_spacing_x, 
                                           detector_spacing_y, 
                                           detector_row_count, 
                                           detector_column_count, 
                                           proj_angles, 
                                           d_source_obj, 
                                           d_obj_detector
                                           )
        
        
        proj_id = astra.create_sino3d_gpu(obj, proj_geom, vol_geom)
    
    
    """
    Possible extension here with more elif: different projection geometries
    #elif ('' == beam):
    """
    
    print("returning obj and proj_id")
    return proj_id



def polychromatic_sinogram(sinogram, material, energy_spectrum, intensity_spectrum, testing = False):
    import numpy as np
    import xraydb
    
    """
    
    Physical explanation:
        Each ray has a width/area accounting for the width/area of one pixel,
        since we are using the strip version. This means whatever dimensions
        the pixel has, a single ray will hit a single detector element (pixel)
    
    """
    
    # Get the attenuation coefficient given the energies and material
    # Takes energies in eV
    u_E = xraydb.mu_elam(material, energy_spectrum*1000)
    
    # Convert u_E from cm^2/g to SI units, m^2/kg
    u_E = u_E / (100*100)*1000

    n_energies = len(energy_spectrum)
    projections_energy = np.zeros( (np.size(sinogram, 0), np.size(sinogram, 1), n_energies) )
    I_all = np.zeros( (np.size(sinogram, 0), np.size(sinogram, 1), n_energies) ) 
    I_0 = np.ones( (np.size(sinogram, 0), np.size(sinogram, 1) ) )
    #T_E = np.ones( (np.size(sinogram, 0), np.size(sinogram, 1), (n_energies,1) ) )
    T_E = np.ones((n_energies,1) ) 
    
    for i in range(n_energies):
        
        
        # Intensity of transmitted light vs incident light, exponential decay
        # due to attenuation coefficient, mu = u_E
        I = intensity_spectrum[i]*I_0 * np.exp(-u_E[i]*sinogram)
        #I = I_0 * np.exp(-u_E[i]*d )
        I_all[:,:,i] = I
        
        T_E[i] = np.min(I/I_0)

    I_total = np.sum(I_all,2)
    I_0_total = I_0*n_energies
    I_0_total = I_0*np.sum(intensity_spectrum)
    transmission = I_total/I_0_total
    absorption = -np.log(transmission)
    poly_sinogram = absorption
    
    

    # Checking if the calculations are done properly and make sense physically
    #if __name__ == '__main__':
    if testing:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title("Transmissions")
        plt.plot(transmission)
        plt.show()
        
        plt.figure()
        plt.title("Absorption")
        plt.plot(absorption)
        plt.show()
    
        plt.figure()
        plt.title("Attenuation as function of energy")
        plt.xlabel("Energy (E)[keV]")
        plt.ylabel("Attenuation (log_10[u_E] ) [m^2/kg]")
        plt.plot(energy_spectrum, np.log10(u_E))
        plt.show()
    
    return poly_sinogram



def polychromatic_sinogram_multiple(sinograms, materials, energy_spectrum, intensity_spectrum, testing = False):
    import numpy as np
    import xraydb
    
    """
    
    Physical explanation:
        Each ray has a width/area accounting for the width/area of one pixel,
        since we are using the strip version. This means whatever dimensions
        the pixel has, a single ray will hit a single detector element (pixel)
    
    """
    
    # Get the attenuation coefficient given the energies and material
    # Takes energies in eV
    u_E = []
    for i in range(len(materials)):
        u_E.append(xraydb.mu_elam(materials[i], energy_spectrum*1000))
        u_E[i] = u_E[i]/ (100*100)*1000     # Convert u_E from cm^2/g to SI units, m^2/kg

    n_energies = len(energy_spectrum)
    
    sinogram = sinograms[0]
    
    projections_energy = np.zeros( (np.size(sinogram, 0), np.size(sinogram, 1), n_energies) )
    I_all = np.zeros( (np.size(sinogram, 0), np.size(sinogram, 1), n_energies) ) 
    I_0 = np.ones( (np.size(sinogram, 0), np.size(sinogram, 1) ) )
    #T_E = np.ones( (np.size(sinogram, 0), np.size(sinogram, 1), (n_energies,1) ) )
    T_E = np.ones((n_energies,1) ) 
    
    #d = sinogram       # units are [m]
    
    for i in range(n_energies):
        
        sinogram_combined = np.zeros( (np.size(sinograms[0],0), np.size(sinograms[0],1)) )
        for j in range(len(sinograms)):
            sinogram_combined += -sinograms[j]*u_E[j][i]
            #sinogram_combined *= (10**j)
        
        
        # Intensity of transmitted light vs incident light, exponential decay
        # due to attenuation coefficient, mu = u_E
        
        
        I = intensity_spectrum[i]*I_0 * np.exp(sinogram_combined)
        I_all[:,:,i] = I
        
        T_E[i] = np.min(I/I_0)

    I_total = np.sum(I_all,2)
    I_0_total = I_0*n_energies
    I_0_total = I_0*np.sum(intensity_spectrum)
    transmission = I_total/I_0_total
    absorption = -np.log(transmission)
    poly_sinogram = absorption
    

    # Checking if the calculations are done properly and make sense physically
    #if __name__ == '__main__':
    if testing:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title("Transmissions")
        plt.plot(transmission)
        plt.show()
        
        plt.figure()
        plt.title("Absorption")
        plt.plot(absorption)
        plt.show()
    
        plt.figure()
        plt.title("Attenuation as function of energy")
        plt.xlabel("Energy (E)[keV]")
        plt.ylabel("Attenuation (log_10[u_E] ) [m^2/kg]")
        for j in range(len(sinograms)):
            plt.plot(energy_spectrum, np.log10(u_E[j]))
        plt.show()
    
    return poly_sinogram






if __name__ == "__main__":
    import numpy as np
    import astra
    
    from create_obj import default_obj_cfg, create_obj
    
    # Lab setup including detection volume, detector dimensions, sample dimensions
    # Angles of scanning by theoretical number of angles needed
    
    # Square detector
    n_pixels = 200
    n_pixels_x = n_pixels
    n_pixels_y = n_pixels

    dim = 2
    n_angles = round(n_pixels*np.pi/2)                    # Theoretical value for needed angles to create good reconstruction
    proj_angles = np.linspace(-np.pi, np.pi, 2*n_angles)  # Angles of projection
    
    # Getting a circular object to test on
    
    obj_dim, centers, shapes, radii = default_obj_cfg(n_pixels, dim)
    
    # Changing to squre for testing
    #shapes[0] = 'square'
    
    obj = create_obj(obj_dim, centers, shapes, radii)
    
    
    # Proper scaled shape to fit in the cone/fan beam
    d_source_obj = n_pixels_x/2
    d_obj_detector = n_pixels_x/2
    
    beam = 'parallel'
    proj_id = create_projector(obj, d_source_obj, d_obj_detector, [n_pixels_x, n_pixels_y], proj_angles, beam)
    
    # Generate sinogram
    sinogram_id, sinogram = astra.create_sino(obj, proj_id)
    sinogram = np.transpose(sinogram)
    
    ## X-ray source definitions yielding a polychromatic beam and a beam hardened sinogram
    material = 'C'
    target_material = 'W'
    filter_material = 'Al'
    
    # Change filter thickness to 0 no matter which filter and you cancel it
    filter_thickness = 4E-3 #m
    target_thickness = 12E-3 #m?
    acceleration_voltage = 80E3 #eV
    
    # Other settings used in the lab
    current = 667E-6 #A
    exposure_time = 500E-3 #s
    voxel_size = 132E-6 #m
    n_angles_exp = 1570 #or 360
    
    
    ## SpekPy to generate the X-ray source Spectrum
    import spekpy
    
    spek_spectrum= spekpy.Spek(kvp = acceleration_voltage/1000, # eV to keV
                               th  = target_thickness*1000 # m to mm
                               ) # Create a spectrum, energy in keV
    spek_spectrum.filter(filter_material, # Al often used
                         filter_thickness*1000 # m to mm
                         ) # Filter the spectrum, thickness in mm?

    energy_spectrum, intensity_spectrum = spek_spectrum.get_spectrum(edges=True) # Get the spectrum
    
    #poly_sinogram = polychromatic_sinogram(sinogram, target_material, filter_material, filter_thickness, material, acceleration_voltage)
    poly_sinogram = polychromatic_sinogram(sinogram, material, energy_spectrum, intensity_spectrum)
    print("converted sino to poly")
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    if (dim == 2):
        plt.imshow(obj)
    elif (dim == 3):
        
        # Just a slice of the 3d obj
        plt.imshow(obj[:,:,round(np.shape(obj)[2]/2)])
    plt.colorbar()
    plt.title("Original Shape")
    plt.show()
    
    
    plt.figure()
    plt.imshow(sinogram)
    plt.colorbar()
    plt.show()
    
    plt.plot(energy_spectrum, intensity_spectrum) # Plot the spectrum
    plt.xlabel('Energy [keV]')
    plt.ylabel('Fluence per mAs per unit energy [photons/cm2/mAs/keV]')
    plt.title('X-ray source spectrum')
    plt.show()
    
    plt.figure()
    plt.imshow(poly_sinogram)
    plt.colorbar()
    plt.show()
    
    
    ## 2D Reconstruction
    [rec_id, rec] = astra.creators.create_reconstruction("FBP", proj_id, np.transpose(sinogram))
    plt.figure()
    plt.imshow(rec)
    plt.colorbar()
    plt.title("Reconstruction")
    plt.show()

    plt.figure() 
    plt.plot( rec[ round( np.size( rec,0 )/2 ),: ]), plt.title('Reconstruction profile')

    ## 2D Reconstruction of Beam Hardened

    # Transposing the sinogram back to astras way of expression
    [rec_harden_id, rec_harden] = astra.creators.create_reconstruction("FBP", proj_id, np.transpose(poly_sinogram) )
    #rec_harden = rec_harden/np.max(rec_harden)

    plt.figure()
    plt.imshow(rec_harden)
    plt.colorbar()
    plt.title("Beam Hardened Reconstruction")
    plt.show()

    plt.figure() 
    plt.plot( rec_harden[ round( np.size( rec_harden,0 )/2 ),: ]), plt.title('Beam Hardened Reconstruction profile')
    
    
    