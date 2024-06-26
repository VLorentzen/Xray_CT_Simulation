src.scanning.generate_spectrum_from_source
==============================================

.. function:: src.scanning.generate_spectrum_from_source(acceleration_voltage, target_thickness, filter_material, filter_thickness)

   Generate an X-ray source spectrum using SpekPy.

   This function uses the SpekPy library to generate an X-ray source spectrum based on the given parameters for acceleration voltage, target thickness, filter material, and filter thickness. The generated spectrum consists of energy and intensity components.

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