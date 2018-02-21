import numpy as np
import pickle
from image import HyperCube

''' This file contains utility functions that don't necessarily belong to the class
    hypercube, as they may not act on a single instantiation of the hypercube class '''

def spectral_correlation(cube1, cube2):
    '''
    Params: Two hypercubes from which the spectral correlation of two different
            regions will be compared.
    '''

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


def spectral_info_divergence(cube1, cube2):
    cube1.select_region('Select Region to Compare', cube1.collect_spectra)
    cube2.select_region('Select Region to Compare', cube2.collect_spectra)

    cube1_spectra = np.clip(np.array(cube1.spectra), 0.01, None)
    cube2_spectra = np.clip(np.array(cube2.spectra), 0.01, None)

    cube1_probability = [wavelength/np.sum(cube1_spectra) for wavelength in cube1_spectra]
    cube2_probability = [wavelength/np.sum(cube2_spectra) for wavelength in cube2_spectra]

    n = len(cube1_probability)

    entropy_xy = np.sum([cube1_probability[i] * np.log2(cube2_probability[i]/cube1_probability[i])
                        for i in range(n)])

    entropy_yx = np.sum([cube2_probability[i] * np.log2(cube1_probability[i]/cube2_probability[i])
                        for i in range(n)])

    return entropy_xy + entropy_yx


def create_spectra_list(image_names):
    """
    Given a list of image names/locations, return a
    list containing the average spectra of each image.
    """

    average_spectra = []

    for image in image_names:

        img = HyperCube(image)
        img.fix_image()
        img.select_region('Select Region', img.set_average_spectra)

        spectra = img.get_average_spectra()
        average_spectra.append(spectra)

    return average_spectra

def pickle_a_day(produce_type, image, age):
    """
    Given an image name, and the age of the produce
    (in days) when the image was taken, calibrate the
    image for darkness and reflectance, select five
    regions from the image, find the average reflectances
    for each, pickling each average reflectance array as
    inputs and the age as the target.

    Specify the type of procude by spelling out the name
    of the produce in full, using lowercase letters, and 
    underscores for spaces (if needed).
    """

    img = HyperCube(image)
    img.fix_image()

    try:
        produce_spectras = pickle.load( open( produce_type+".p", "rb" ))

    except (OSError, IOError) as e:
        produce_spectras = []

    for _ in range(5):
        img.select_region('Select Region', img.set_average_spectra)

        spectra = np.append(img.get_average_spectra(), age) #add age as target
        produce_spectras.append(spectra)

    pickle.dump(produce_spectras, open( produce_type+".p", "wb" ) )

def clear_pickle(produce_type):
    """
    Clear the contents of a pickle file for a
    given type of produce.
    """

    pickle.dump([], open( produce_type+".p", "wb" ) )