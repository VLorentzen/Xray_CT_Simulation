src.scanning.polychromatic_sinogram
=======================================

.. function:: src.scanning.polychromatic_sinogram(sinogram, materials, energy_spectrum, intensity_spectrum, testing=False)

    Convert a monochromatic sinogram to a polychromatic sinogram using given material properties and energy spectrum.

    This function takes a monochromatic sinogram and simulates the polychromatic effect by considering the energy-dependent attenuation coefficients of the materials. The output is a polychromatic sinogram.

    :param sinogram: The input sinogram, which can be a 2D or 3D numpy array.
    :type sinogram: numpy.ndarray
    :param materials: List of materials through which the X-rays pass.
    :type materials: list of str
    :param energy_spectrum: Array of energy values in keV.
    :type energy_spectrum: numpy.ndarray
    :param intensity_spectrum: Array of intensity values corresponding to the energy spectrum.
    :type intensity_spectrum: numpy.ndarray
    :param testing: If True, additional testing outputs are enabled (default is False).
    :type testing: bool, optional

    :returns: The polychromatic sinogram.
    :rtype: numpy.ndarray

    .. note::
        The xraydb and matplotlib libraries must be installed to use this function.

    **Examples**

    .. code-block:: python

        >>> import numpy as np
        >>> sinogram = np.ones((512, 180))  # Example 2D sinogram
        >>> materials = ['Al']
        >>> energy_spectrum = np.linspace(20, 120, 100)  # Energy from 20 keV to 120 keV
        >>> intensity_spectrum = np.ones(100)  # Uniform intensity spectrum
        >>> poly_sinogram = polychromatic_sinogram(sinogram, materials, energy_spectrum, intensity_spectrum)
        >>> print(poly_sinogram)