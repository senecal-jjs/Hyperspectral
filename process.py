from image import HyperCube
import pickle
import numpy as np
import utils


if __name__ == '__main__':
    #img_1 = HyperCube('Banana_1_Day1.bil')
    #img_1.fix_image()

    #img_1.select_region('Select Region to Plot', img_1.plot_average_spectra)
    
    #utils.pickle_a_day('banana', 'Banana_1_Day15.bil', 15)

    try:
        produce_spectras = pickle.load( open( "banana.p", "rb" ))

    except (OSError, IOError) as e:
        produce_spectras = []

    for spectra in produce_spectras:
    	print spectra[290]