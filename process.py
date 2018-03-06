from image import HyperCube
import pickle
import numpy as np
import utils


if __name__ == '__main__':
    img_1 = HyperCube('Tomato_1_Day6.bil')
    #img_1.fix_image()
    #grayscale_image = img_1.to_grayscale("Gray Banana")

    #black_or_white = utils.to_black_and_white(img_1, 0.35, 0.85)
    #utils.canny_edge_detection(black_or_white)

    utils.detect_produce(img_1)
    #utils.detect_spectralon(img_1)

    #img_1.select_region('Select Region to Plot', img_1.plot_average_spectra)
    
    #utils.pickle_a_day('banana', 'Banana_1_Day15.bil', 15)

    """try:
        produce_spectra = pickle.load( open( "tomato.p", "rb" ))

    except (OSError, IOError) as e:
        produce_spectra = []

    reflectances = [item[:290] for item in produce_spectra]
    labels = [int(item[290]) for item in produce_spectra]

    #utils.feature_selection(reflectances, 8)

    #new_data = utils.principal_components(reflectances, 2)
    new_data = utils.kernel_pca(reflectances, 2)
    utils.plot_components(new_data, labels)"""

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

    #new_data = utils.principal_components(paper_wavelengths, 2)
    new_data = utils.kernel_pca(paper_wavelengths, 2)
    utils.plot_components(new_data, labels)"""

    #utils.pls_regression(reflectances, labels)

    """tomato_wavelengths = []
    for spectrum in produce_spectra:
        temp = []
        temp.append(spectrum[146])
        temp.append(spectrum[147])
        temp.append(spectrum[144])
        temp.append(spectrum[145])
        temp.append(spectrum[148])
        temp.append(spectrum[150])
        temp.append(spectrum[143])
        temp.append(spectrum[149])
        temp = np.array(temp)
        tomato_wavelengths.append(temp)

    #new_data = utils.principal_components(tomato_wavelengths, 2)
    new_data = utils.kernel_pca(tomato_wavelengths, 2)
    utils.plot_components(new_data, labels)"""