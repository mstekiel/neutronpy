from setuptools import setup

setup(name='neutronpy',
    version='2.0.0',
    description = "NeutronPy is a collection of commonly used tools aimed at facilitating the analysis of neutron scattering data. NeutronPy is built primarily using the numpy and scipy python libraries, with a translation of ResLib 3.4c (MatLab) routines for Instrument resolution calculations.",
    url='https://github.com/mstekiel/neutronpy',
    author='Michal Stekiel',
    author_email='michal.stekiel@gmail.com',
    packages=['neutronpy'],
    python_requires='>=3.9',
    # install_requires=[
    #     "numpy>=1.10",
    #     "scipy>=1.0",
    #     "matplotlib>=3.5",
    #     "h5py>=3.1.0",
    #     "lmfit>=1.0",
    #     "brille"
    # ],
    extras_require={
        "dev": ['sphinx==5.0.2', 'pandas==1.4.4'],
    },
    include_package_data=True,
    zip_safe=False)