from image import HyperCube
import numpy as np


if __name__ == '__main__':
    img_1 = HyperCube('RedDelicious_1_Day15.bil')
    coordinates = img_1.get_img_coords(img_1.image, 'Select Calibration Region!')
