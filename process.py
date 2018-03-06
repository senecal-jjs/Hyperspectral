from image import HyperCube
import pickle
import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy 


if __name__ == '__main__':
    # cube1 = HyperCube('YukonGold_1_Day6.bil')
    # cube1.fix_image()
    # cube2 = HyperCube('Tomato_1_Day14.bil')
    # cube2.fix_image()

    #cube1.select_region('Select Region to Plot', cube1.plot_average_spectra)
    # cube2.select_region('Select Region to Plot', cube2.plot_average_spectra)

    # cube1.select_region('Select Region to Compare', cube1.collect_spectra)
    # cube2.select_region('Select Region to Compare', cube2.collect_spectra)

    # print(utils.spectral_info_divergence(cube1.spectra, cube2.spectra))

    utils.pickle_a_day('potato1', 'YukonGold_1_Day1.bil', 1, num_regions=3)
    utils.pickle_a_day('potato1', 'YukonGold_1_Day2.bil', 2, num_regions=3)
    utils.pickle_a_day('potato1', 'YukonGold_1_Day3.bil', 3, num_regions=3)
    utils.pickle_a_day('potato1', 'YukonGold_1_Day6.bil', 6, num_regions=3)
    utils.pickle_a_day('potato1', 'YukonGold_1_Day7.bil', 7, num_regions=3)
    utils.pickle_a_day('potato1', 'YukonGold_1_Day10.bil', 10, num_regions=3)
    utils.pickle_a_day('potato1', 'YukonGold_1_Day14.bil', 14, num_regions=3)
    # plt.plot(reflectances[0])
    # plt.plot(reflectances[5])
    # plt.plot(reflectances[10])
    # plt.plot(reflectances[15])
    # plt.plot(reflectances[20])
    # plt.plot(reflectances[25])
    # plt.show()

    # 5 different measuresments from Day 0
    # baseline_reflectance = reflectances[0:5]
    # divergence = np.zeros([10, 6])
    # ave_divergence = np.zeros(10)
    # days = np.array([1,2,3,5,6,7,9,10,13,15])
    # for i in range(0, 50, 5):
    #     r1, r2, r3, r4, r5 = i, i+1, i+2, i+3, i+4
    #     divergence[i/5,0] = days[i/5]
    #     divergence[i/5,1] = utils.spectral_info_divergence(baseline_reflectance[0], reflectances[r1])
    #     divergence[i/5,2] = utils.spectral_info_divergence(baseline_reflectance[1], reflectances[r2])
    #     divergence[i/5,3] = utils.spectral_info_divergence(baseline_reflectance[2], reflectances[r3])
    #     divergence[i/5,4] = utils.spectral_info_divergence(baseline_reflectance[3], reflectances[r4])
    #     divergence[i/5,5] = utils.spectral_info_divergence(baseline_reflectance[4], reflectances[r5])
    #     ave_divergence[i/5] = np.mean(divergence[i/5,:])

    # df = pd.DataFrame({"day":[1,2,3,5,6,7,9,10,13,15], "val": ave_divergence})
    # df.set_index("day",drop=True,inplace=True)
    # df.plot.bar(title="Average Spectral Information Divergence", rot=0, legend=False)
    # plt.xlabel("Days Since Purchase")

    # df2 = pd.DataFrame(divergence, columns=['Day','Location A', 'Location B', 'Location C', 'Location D', 'Location E'])
    # df2.set_index("Day", drop=True, inplace=True)
    # print(df2)
    # df2.plot.bar(title="Spectral Information Divergence", rot=0)
    # plt.xlabel("Days Since Purchase")
    # plt.show()

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

    #utils.pls_regression(reflectances, labels)
