from image import HyperCube
import numpy as np


if __name__ == '__main__':
    img_1 = HyperCube('Banana_1_Day1.bil')
    img_1.fix_image()

    img_1.display_rgb('Calibrated Image!')
