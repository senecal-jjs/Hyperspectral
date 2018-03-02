from image import HyperCube
import pickle
import numpy as np
import utils
from sklearn.preprocessing import normalize


if __name__ == '__main__':
    #img_1 = HyperCube('Banana_1_Day1.bil')
    #img_1.fix_image()

    #img_1.select_region('Select Region to Plot', img_1.plot_average_spectra)
    
    #utils.pickle_a_day('banana', 'Banana_1_Day15.bil', 15)

    try:
        produce_spectra = pickle.load( open( "banana.p", "rb" ))

    except (OSError, IOError) as e:
        produce_spectra = []


    reflectances = [item[:290] for item in produce_spectra]
    labels = [int(item[290]) for item in produce_spectra]

    #reflectances = normalize(reflectances)

    utils.pearson_coefficient_features(reflectances, labels, 8)
    #print pearson_coef

    #utils.feature_selection(reflectances, 8)

    #new_data = utils.principal_components(reflectances, 2)
    #new_data = utils.kernel_pca(reflectances, 2)
    #utils.plot_components(new_data, labels)


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

    #new_data = utils.principal_components(paper_wavelengths, 2)
    new_data = utils.kernel_pca(paper_wavelengths, 2)
    utils.plot_components(new_data, labels)"""

    #utils.pls_regression(reflectances, labels)

    """tomato_wavelengths = []
    for spectrum in produce_spectra:
        temp = []
        temp.append(spectrum[269])
        temp.append(spectrum[274])
        temp.append(spectrum[271])
        temp.append(spectrum[277])
        temp.append(spectrum[283])
        temp.append(spectrum[268])
        temp.append(spectrum[270])
        temp.append(spectrum[267])
        temp = np.array(temp)
        tomato_wavelengths.append(temp)

    #new_data = utils.principal_components(tomato_wavelengths, 2)
    new_data = utils.kernel_pca(tomato_wavelengths, 2)
    utils.plot_components(new_data, labels)"""

    """banana_wavelengths = []
    for spectrum in produce_spectra:
        temp = []
        temp.append(spectrum[289])
        temp.append(spectrum[72])
        temp.append(spectrum[92])
        temp.append(spectrum[93])
        temp.append(spectrum[94])
        temp.append(spectrum[95])
        temp.append(spectrum[96])
        temp.append(spectrum[97])
        temp = np.array(temp)
        banana_wavelengths.append(temp)

    #new_data = utils.principal_components(banana_wavelengths, 2)
    new_data = utils.kernel_pca(banana_wavelengths, 2)
    utils.plot_components(new_data, labels) """