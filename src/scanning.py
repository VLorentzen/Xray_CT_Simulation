"""
Goals
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
    """
    Create a projector using the ASTRA Toolbox.

    This function sets up the volume and projection geometries using the ASTRA Toolbox and initializes the projector based on the specified beam type. Supported geometries include 2D parallel, 2D fan beam, 3D parallel, and 3D cone beam projections.

    Parameters
    ----------
    obj : object
        The object to be projected. This is typically a 2D or 3D numpy array representing the object.
    d_source_obj : float
        Distance from the X-ray source to the object.
    d_obj_detector : float
        Distance from the object to the detector.
    n_pixels : int or tuple of int
        Number of pixels in the detector. For 2D, this is an integer. For 3D, this is a tuple (n_pixels_y, n_pixels_x).
    proj_angles : array-like
        Array of projection angles (in radians).
    beam : str
        Type of beam used for projection. Supported values are 'parallel', 'fanflat', 'parallel3d', 'cone'.
    testing : bool, optional
        If True, additional testing outputs are enabled (default is False).

    Returns
    -------
    int
        The ID of the created projector.

    Raises
    ------
    ValueError
        If the dimensions of `n_pixels` are greater than 3 or if an unsupported beam type is provided.

    Notes
    -----
    The ASTRA Toolbox must be installed to use this function. For more information on ASTRA, visit:
    https://astra-toolbox.com/docs/proj2d.html

    Examples
    --------
    >>> import numpy as np
    >>> obj = np.ones((512, 512))  # Example 2D object
    >>> d_source_obj = 500.0
    >>> d_obj_detector = 500.0
    >>> n_pixels = 512
    >>> proj_angles = np.linspace(0, np.pi, 180, endpoint=False)
    >>> beam = 'parallel'
    >>> proj_id = create_projector(obj, d_source_obj, d_obj_detector, n_pixels, proj_angles, beam)
    >>> print(proj_id)
    """
    
    
    try:
        dim = len(n_pixels)
    except:
        dim = 2
        n_pixels_y = n_pixels
        n_pixels_x = 0
    else:
        if (dim > 3):
            raise ValueError("Dimensions unphysical, must be 2D or 3D")
        n_pixels_y = n_pixels[0]
        n_pixels_x = n_pixels[1]


    import astra
    
    d_source_detector = int(round(d_source_obj + d_obj_detector))
    
    # Define volume geometry using ASTRA
    if (dim == 2):
        vol_geom = astra.create_vol_geom(n_pixels_y, d_source_detector)
        
    elif(dim == 3):
        vol_geom = astra.create_vol_geom(n_pixels_y, d_source_detector, n_pixels_x)

    
    ## DETECTOR DEFINITIONS
    # 2 dimensions are max, x,y. Basic step size used as 1.0
    
    # 1st dimension of detector in y direction
    detector_spacing_y = 1.0
    detector_row_count = n_pixels_y
    
    # 2nd dimension of detector in x direction
    detector_spacing_x = 1.0
    detector_column_count = n_pixels_x
    
    
    ## Create Projection through Astra ##
    """ 
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
        proj_geom = astra.create_proj_geom('parallel', detector_spacing_y, detector_row_count, proj_angles)
        
        # Supported: line, strip, linear
        proj_id = astra.create_projector('strip', proj_geom, vol_geom)
    
    # 2d Fan(flat) beam projection initialization
    elif ('fanflat' == beam):    
        proj_geom = astra.create_proj_geom('fanflat', detector_spacing_y, detector_row_count, proj_angles, d_source_obj, d_obj_detector)
        
        # Supported: line_fanflat, strip_fanflat
        proj_id = astra.create_projector('strip_fanflat', proj_geom, vol_geom)
    
    # 3d Parallel projection initialization
    elif ('parallel3d' == beam):
        proj_geom = astra.create_proj_geom('parallel3d', 
                                           detector_spacing_y, 
                                           detector_spacing_x, 
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
                                           detector_spacing_y, 
                                           detector_spacing_x, 
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

def generate_spectrum_from_source(acceleration_voltage, target_thickness, filter_material, filter_thickness):
    """
    Generate an X-ray source spectrum using SpekPy.

    This function uses the SpekPy library to generate an X-ray source spectrum based on the given parameters
    for acceleration voltage, target thickness, filter material, and filter thickness. The generated spectrum
    consists of energy and intensity components.

    Parameters
    ----------
    acceleration_voltage : float
        The acceleration voltage of the X-ray source in electron volts (eV).
    target_thickness : float
        The thickness of the X-ray source target in meters (m).
    filter_material : str
        The material used for filtering the X-ray spectrum (e.g., 'Al' for Aluminum).
    filter_thickness : float
        The thickness of the filter material in meters (m).

    Returns
    -------
    tuple
        A tuple containing two elements:
        - energy_spectrum (numpy.ndarray): The energy values of the spectrum (in keV).
        - intensity_spectrum (numpy.ndarray): The intensity values of the spectrum.

    Examples
    --------
    >>> energy_spectrum, intensity_spectrum = generate_spectrum_from_source(
    ...     acceleration_voltage=100000,
    ...     target_thickness=0.01,
    ...     filter_material='Al',
    ...     filter_thickness=0.005
    ... )
    >>> print(energy_spectrum)
    >>> print(intensity_spectrum)

    Notes
    -----
    The SpekPy library must be installed to use this function. For more information on SpekPy, visit:
    https://spekpy.github.io/
    """
    
    ## SpekPy to generate the X-ray source Spectrum
    import spekpy
    
    spek_spectrum= spekpy.Spek(kvp = acceleration_voltage/1000, # eV to keV
                               th  = target_thickness*1000 # m to mm
                               ) # Create a spectrum, energy in keV
    spek_spectrum.filter(filter_material, # Al often used
                         filter_thickness*1000 # m to mm
                         ) # Filter the spectrum, thickness in mm?

    energy_spectrum, intensity_spectrum = spek_spectrum.get_spectrum(edges=True) # Get the spectrum
    
    return energy_spectrum, intensity_spectrum


def polychromatic_sinogram(sinogram, materials, energy_spectrum, intensity_spectrum, testing = False):
    """
    Convert a monochromatic sinogram to a polychromatic sinogram using given material properties and energy spectrum.

    This function takes a monochromatic sinogram and simulates the polychromatic effect 
    by considering the energy-dependent attenuation coefficients of the materials. The output is a polychromatic sinogram.

    Parameters
    ----------
    sinogram : numpy.ndarray
        The input sinogram, which can be a 2D or 3D numpy array.
    materials : list of str
        List of materials through which the X-rays pass.
    energy_spectrum : numpy.ndarray
        Array of energy values in keV.
    intensity_spectrum : numpy.ndarray
        Array of intensity values corresponding to the energy spectrum.
    testing : bool, optional
        If True, additional testing outputs are enabled (default is False).

    Returns
    -------
    numpy.ndarray
        The polychromatic sinogram.

    Notes
    -----
    The xraydb and matplotlib libraries must be installed to use this function.

    Examples
    --------
    >>> import numpy as np
    >>> sinogram = np.ones((512, 180))  # Example 2D sinogram
    >>> materials = ['Al']
    >>> energy_spectrum = np.linspace(20, 120, 100)  # Energy from 20 keV to 120 keV
    >>> intensity_spectrum = np.ones(100)  # Uniform intensity spectrum
    >>> poly_sinogram = polychromatic_sinogram(sinogram, materials, energy_spectrum, intensity_spectrum)
    >>> print(poly_sinogram)
    """
    
    import numpy as np
    import xraydb
    import matplotlib.pyplot as plt
    """
    
    Physical explanation:
        Each ray has a width/area accounting for the width/area of one pixel,
        since we are using the strip version. This means whatever dimensions
        the pixel has, a single ray will hit a single detector element (pixel)
    
    """
    # Checking if one or multiple sinograms
    try:
        n_sinogram_pixels = np.size(sinogram[0], 0)
        n_sinogram_projections = np.size(sinogram[0], 1)
        n_sinogram = len(sinogram)
    except:
        n_sinogram_pixels = np.size(sinogram, 0)
        n_sinogram_projections = np.size(sinogram, 1)
        n_sinogram = 1
        sinogram = [sinogram]
        materials = [materials]
    
    # Get the attenuation coefficient given the energies and material
    # Takes energies in eV
    attenuation_coefficient = []
    for i in range(len(materials)):
        attenuation_coefficient.append(xraydb.mu_elam(materials[i], energy_spectrum*1000))
        attenuation_coefficient[i] = attenuation_coefficient[i]/ (100*100)*1000     # Convert u_E from cm^2/g to SI units, m^2/kg

    n_energies = len(energy_spectrum)
    
    
    intensity_all = np.zeros( (n_sinogram_pixels, n_sinogram_projections, n_energies) ) 
    intensity_0 = np.ones( (n_sinogram_pixels, n_sinogram_projections) )

    for i in range(n_energies):
        sinogram_combined = np.zeros( (n_sinogram_pixels, n_sinogram_projections) )
        
        # Intensity of transmitted light vs incident light, exponential decay
        # due to attenuation coefficient, mu = u_E
        intensity = intensity_spectrum[i]*intensity_0
        
        for j in range(n_sinogram):
            sinogram_combined -= sinogram[j]*attenuation_coefficient[j][i]
            
        # Update the intensities according to Beer-Lambert Law
        intensity *= np.exp(sinogram_combined)
        intensity_all[:,:,i] = intensity

    intensity_total = np.sum(intensity_all,2)
    intensity_0_total = intensity_0*n_energies
    intensity_0_total = intensity_0*np.sum(intensity_spectrum)
    
    transmission = intensity_total/intensity_0_total
    absorption = -np.log(transmission)
    poly_sinogram = absorption
    
    plt.figure()
    plt.title("Attenuation Coefficient vs Energy")
    plt.xlabel("Energy (E)[keV]")
    plt.ylabel("Attenuation (log_10[u_E] ) [m^2/kg]")
    for j in range(n_sinogram):
        plt.plot(energy_spectrum, np.log10(attenuation_coefficient[j]))
    plt.legend(materials)
    plt.show()


    # Checking if the calculations are done properly and make sense physically
    if testing:
        
        plt.figure()
        plt.title("Transmissions")
        plt.plot(transmission)
        plt.show()
        
        plt.figure()
        plt.title("Absorption")
        plt.plot(absorption)
        plt.show()
        
    print("converted sinogram from monochromatic to polychromatic")
    return poly_sinogram


if __name__ == "__main__":
    import numpy as np
    import astra
    
    from create_obj import default_obj_cfg, create_obj
    
    # Lab setup including detection volume, detector dimensions, sample dimensions
    # Angles of scanning by theoretical number of angles needed
    
    # Square detector
    n_pixels = 200
    n_pixels_y = n_pixels
    n_pixels_x = n_pixels

    dim = 2
    n_angles = round(n_pixels*np.pi/2)                    # Theoretical value for needed angles to create good reconstruction
    proj_angles = np.linspace(-np.pi, np.pi, 2*n_angles)  # Angles of projection
    
    # Getting a circular object to test on
    obj_dim, centers, shapes, radii = default_obj_cfg(n_pixels, dim)
    obj = create_obj(obj_dim, centers, shapes, radii)
    
    
    # Proper scaled shape to fit in the cone/fan beam
    d_source_obj = n_pixels_y/2
    d_obj_detector = n_pixels_y/2
    
    beam = 'parallel'
    proj_id = create_projector(obj, d_source_obj, d_obj_detector, [n_pixels_y, n_pixels_x], proj_angles, beam)
    
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
    
    # Using function in this file to generate a spectrum given the settings
    energy_spectrum, intensity_spectrum = generate_spectrum_from_source(acceleration_voltage, 
                                                                        target_thickness, 
                                                                        filter_material, 
                                                                        filter_thickness
                                                                        )
    
    poly_sinogram = polychromatic_sinogram(sinogram, material, energy_spectrum, intensity_spectrum)
    
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

    plt.figure()
    plt.imshow(rec_harden)
    plt.colorbar()
    plt.title("Beam Hardened Reconstruction")
    plt.show()

    plt.figure() 
    plt.plot( rec_harden[ round( np.size( rec_harden,0 )/2 ),: ]), plt.title('Beam Hardened Reconstruction profile')
    
    
    