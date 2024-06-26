import spekpy as sp # Import SpekPy
import matplotlib.pyplot as plt # Import library for plotting

acceleration_voltage = 80 #keV
filter_material = 'Al'
filter_thickness = 1 #mm

spek_spectrum= sp.Spek(kvp = acceleration_voltage, th = 12) # Create a spectrum
spek_spectrum.filter(filter_material, filter_thickness) # Filter the spectrum

hvl = spek_spectrum.get_hvl1() # Get the 1st half-value-layer
print(hvl, 'mm') # Print value to screen

energy_spectrum, intensity_spectrum = spek_spectrum.get_spectrum(edges=True) # Get the spectrum

plt.plot(energy_spectrum, intensity_spectrum) # Plot the spectrum
plt.xlabel('Energy [keV]')
plt.ylabel('Fluence per mAs per unit energy [photons/cm2/mAs/keV]')
plt.title('X-ray source spectrum')
plt.show()