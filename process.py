from image import HyperCube
import numpy as np


if __name__ == '__main__':
    img_1 = HyperCube('RedDelicious_1_Day18.bil')
    img_1.fix_image()

    img_1.select_region('Select Region to Plot', img_1.plot_average_spectra)