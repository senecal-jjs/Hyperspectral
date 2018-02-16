import numpy as np

''' This file contains utility functions that don't necessarily belong to the class
    hypercube, as they may not act on a single instantiation of the hypercube class '''

def spectral_correlation(cube1, cube2):
    cube1.select_region('Select Region to Compare', cube1.collect_spectra)
    cube2.select_region('Select Region to Compare', cube2.collect_spectra)

    cube1_spectra = np.array(cube1.spectra)
    cube2_spectra = np.array(cube2.spectra)
    n = len(cube1_spectra)

    numerator = (n * np.sum(cube1_spectra * cube2_spectra)
                - (np.sum(cube1_spectra) * np.sum(cube2_spectra)))

    denominator = np.sqrt((n * np.sum(cube1_spectra**2) - np.sum(cube1_spectra)**2)
                        * (n * np.sum(cube2_spectra**2) - np.sum(cube2_spectra)**2))

    return numerator/denominator
