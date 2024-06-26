# Xray-Lab-Imaging-Simulation
The purpose of this repository is to provide a simulation tool for X-ray laboratory Computed Tomography (CT) Imaging. Given a digital object (2D or 3D), the tool can perform simulated scans using relevant X-ray Absorption Imaging parameters including:
- Specimen material elements
- Acceleration voltage
- Object positioning in the scanner
- Filters applied to the radiation from the target.

This tool highlights the artifacts that arise due to poly-chromatic beams. This tool is made in part to enhance the course "47209: 3D Imaging, Analysis and Modelling" at the Technical University of Denmark, DTU Energy.

# Authors
Victor Lorentzen, Student at Technical University of Denmark, DTU Physics

# Supervisors
DTU Energy:
Peter Stanley Jørgensen,
Salvatore di Angelo

# Used existing tools and packages
- Astra Toolbox https://github.com/astra-toolbox/astra-toolbox
    Provides the projections, makes sinograms and generates reconstructions. Main powerhouse driving the scripts
- SpekPy https://bitbucket.org/spekpy/spekpy_release/wiki/Home
    Simulates a realistic in lab intensity vs energy spectrum
- XrayDB https://xraypy.github.io/XrayDB/python.html
    Provides theoretical values such as the attenuation coefficients of different elements
- standard python libraries such as matplotlib, numpy ... Check the file "requirements.yml" for more info


# Get started using the modules and demos
It is recommended to install a version of anaconda and install the necessary packages through the an anaconda terminal such as the Anaconda Prompt.
This can be done by:
1. Installing the Anaconda Navigator https://docs.anaconda.com/anaconda/install/windows/
2. Download the file "requirements.yml" and remember the path to its location
3. Open Anaconda Prompt and create a new environment by typing following into the command line: conda env create -n xray_ct_sim --file requirements.yml
3. Make sure the entire path is included, could be something like: C:\Users\XXXX\Documents\Xray-Lab-Imaging-Simulation\requirements.yml. Remember to use "" around the path if there are any spaces
4. Activate the environment from the Anaconda Prompt by using the command: conda activate xray_ct_sim
5. Open Spyder by typing spyder in the command line and go the folder where the repository located.
6. Go crazy!

