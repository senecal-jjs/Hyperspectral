from image import HyperCube
import pickle
import numpy as np
import utils


if __name__ == '__main__':
    img_1 = HyperCube('Banana_1_Day1.bil')
    #img_1.fix_image()

    #img_1.select_region('Select Region to Plot', img_1.plot_average_spectra)
    
    #utils.pickle_a_day('banana', 'Banana_1_Day15.bil', 15)

    try:
        produce_spectra = pickle.load( open( "banana.p", "rb" ))

    except (OSError, IOError) as e:
        produce_spectra = []

    reflectances = [item[:290] for item in produce_spectra]
    labels = [int(item[290]) for item in produce_spectra]


    #new_data = utils.principal_components(reflectances, 2)
    #utils.plot_components(new_data, labels)

    """print img_1.imager_wavelengths[26]
    print img_1.imager_wavelengths[68]
    print img_1.imager_wavelengths[120]
    print img_1.imager_wavelengths[138]
    print img_1.imager_wavelengths[156]
    print img_1.imager_wavelengths[179]
    print img_1.imager_wavelengths[255]
    print img_1.imager_wavelengths[281]"""

    """paper_wavelengths = []
    for spectrum in produce_spectra:
    	temp = []
    	temp.append(spectrum[26])
    	temp.append(spectrum[68])
    	temp.append(spectrum[120])
    	temp.append(spectrum[138])
    	temp.append(spectrum[156])
    	temp.append(spectrum[179])
    	temp.append(spectrum[255])
    	temp.append(spectrum[281])
    	temp = np.array(temp)
    	paper_wavelengths.append(temp)

    new_data = utils.principal_components(paper_wavelengths, 2)"""
    #utils.plot_components(new_data, labels)

    utils.pls_regression(reflectances, labels)