import gdal
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from matplotlib.widgets  import RectangleSelector


class HyperCube():
    def __init__(self, raw_bil_file):
        self.image = self.convert_bil_to_array(raw_bil_file)

        # Wavelength specific reflectance for the Spectralon calibration panel
        # Maps wavelength (nm) to reflectance value
        # Source: March 8, 2011 calibration certificate for S/N 7A11D-1394 (box 72019)
        self.spectralon_reflectance = { 300 : 0.977,
                                        350 : 0.984,
                                        400 : 0.989,
                                        450 : 0.988,
                                        500 : 0.988,
                                        550 : 0.989,
                                        600 : 0.989,
                                        650 : 0.988,
                                        700 : 0.988,
                                        750 : 0.988,
                                        800 : 0.987,
                                        850 : 0.987,
                                        900 : 0.989,
                                        950 : 0.989,
                                        1000 : 0.989,
                                        1050 : 0.988,
                                        1100 : 0.988,
                                        1150 : 0.988,
                                        1200 : 0.987,
                                        1250 : 0.987,
                                        1300 : 0.986,
                                        1350 : 0.985,
                                        1400 : 0.985,
                                        1450 : 0.986,
                                        1500 : 0.986}

        # The wavelengths of the Pika L imager
        # Note, leave this as an array, not as a tuple, to manipulate it if need be
        # Also, each row is ten wavelengths for use of searching through it

        # General note: band 218 is wavelength 841.45 nm (infrared benchmark)
        #               121 -> 640.936 nm (visible, red)
        #               77  -> 550.613 nm (visible, green)
        #               33  -> 460.29  nm (visible, blue)

        self.imager_wavelengths = np.array([
            387.12, 389.13, 391.13, 393.13, 395.14, 397.14, 399.15, 401.16, 403.17, 405.17,
            407.18, 409.19, 411.21, 413.22, 415.23, 417.25, 419.26, 421.28, 423.29, 425.31,
            427.33, 429.35, 431.37, 433.39, 435.41, 437.43, 439.46, 441.48, 443.51, 445.53,
            447.56, 449.59, 451.62, 453.64, 455.68, 457.71, 459.74, 461.77, 463.8, 465.84,
            467.87, 469.91, 471.95, 473.98, 476.02, 478.06, 480.1, 482.14, 484.19, 486.23,
            488.27, 490.32, 492.36, 494.41, 496.46, 498.5, 500.55, 502.6, 504.65, 506.7,
            508.76, 510.81, 512.86, 514.92, 516.97, 519.03, 521.09, 523.15, 525.21, 527.26,
            529.33, 531.39, 533.45, 535.51, 537.58, 539.64, 541.71, 543.77, 545.84, 547.91,
            549.98, 552.05, 554.12, 556.19, 558.26, 560.34, 562.41, 564.49, 566.56, 568.64,
            570.72, 572.79, 574.87, 576.95, 579.04, 581.12, 583.2, 585.28, 587.37, 589.45,
            591.54, 593.63, 595.71, 597.8, 599.89, 601.98, 604.07, 606.16, 608.26, 610.35,
            612.45, 614.54, 616.64, 618.73, 620.83, 622.93, 625.03, 627.13, 629.23, 631.33,
            633.44, 635.54, 637.65, 639.75, 641.86, 643.96, 646.07, 648.18, 650.29, 652.4,
            654.51, 656.63, 658.74, 660.85, 662.97, 665.08, 667.2, 669.32, 671.44, 673.55,
            675.67, 677.8, 679.92, 682.04, 684.16, 686.29, 688.41, 690.54, 692.66, 694.79,
            696.92, 699.05, 701.18, 703.31, 705.44, 707.57, 709.71, 711.84, 713.98, 716.11,
            718.25, 720.39, 722.53, 724.67, 726.81, 728.95, 731.09, 733.23, 735.38, 737.52,
            739.66, 741.81, 743.96, 746.11, 748.25, 750.4, 752.55, 754.71, 756.86, 759.01,
            761.16, 763.32, 765.47, 767.63, 769.79, 771.95, 774.1, 776.26, 778.42, 780.59,
            782.75, 784.91, 787.08, 789.24, 791.41, 793.57, 795.74, 797.91, 800.08, 802.25,
            804.42, 806.59, 808.76, 810.93, 813.11, 815.28, 817.46, 819.64, 821.81, 823.99,
            826.17, 828.35, 830.53, 832.71, 834.89, 837.08, 839.26, 841.45, 843.63, 845.82,
            848.01, 850.2, 852.39, 854.58, 856.77, 858.96, 861.15, 863.34, 865.54, 867.73,
            869.93, 872.13, 874.32, 876.52, 878.72, 880.92, 883.12, 885.33, 887.53, 889.73,
            891.94, 894.14, 896.35, 898.56, 900.76, 902.97, 905.18, 907.39, 909.6, 911.82,
            914.03, 916.24, 918.46, 920.67, 922.89, 925.11, 927.32, 929.54, 931.76, 933.98,
            936.21, 938.43, 940.65, 942.87, 945.1, 947.33, 949.55, 951.78, 954.01, 956.24,
            958.47, 960.7, 962.93, 965.16, 967.39, 969.63, 971.86, 974.1, 976.34, 978.57,
            980.81, 983.05, 985.29, 987.53, 989.77, 992.02, 994.26, 996.5, 998.75, 1001.0,
            1003.24, 1005.49, 1007.74, 1009.99, 1012.24, 1014.49, 1016.74, 1018.99, 1021.25, 1023.5
        ])



    def convert_bil_to_array(self, raw_bil_file):
        gdal.GetDriverByName('EHdr').Register()
        # Open the .bil file
        raw_image = gdal.Open(raw_bil_file)

        # Read in each spectral band, (there are 240 bands)
        image = np.dstack([self.read_bil_file(raw_image, ii) for ii in range(1, 241)]).astype('float32')

        return image


    ### Read in .bil files as np array
    # rasterband is the above defined spectral_bands
    def read_bil_file(self, img, rasterband):
        band = img.GetRasterBand(rasterband)
        data = band.ReadAsArray()
        return data

    def fix_image(self):
        """
        Perform darkness correction and reflectance calibration.
        """

        self.image = self.dark_correction()
        #self.image = self.calibrate()
        self.select_calibration_panel("Select calibration panel")

        #return calibrated


    '''def calibrate(self):
        #calibration = np.dstack([self.read_bil_file(calibration_bil, ii).astype('float32') for ii in range(1, 241)])
        
        rs = RectangleSelector(ax, line_select_callback,
                       drawtype='box', useblit=False, button=[1], 
                       minspanx=5, minspany=5, spancoords='pixels', 
                       interactive=True)

        self.select_calibration_panel("Select calibration panel")'''

    def calibrate(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

        plt.close()

        panel = self.image[x1:x2,y1:y2,:] #data = dataCube[pointlist[:, 1], pointlist[:, 0], :]


        # Extract the mean panel image
        panel_mean = np.mean(panel, axis=0)

        # Extract Spectralon reflectances for each wavelength
        reflectances = np.array(map(self.get_spectralon_reflectance, self.imager_wavelengths[:240])).astype('float32')


        # Calculate DN -> reflectance correction for each wavelength
        correction = np.mean(reflectances[None,:] / panel_mean, axis=0)
        correction = correction[None,:][None,:] # Broadcast correction to 3d array

        self.image = self.image * correction

    def dark_correction(self):
        """ 
        Using the dark_correction.bil file, correct for darkness
        in the image. Assumes file to be in same directory as code.
        """

        dark_image = HyperCube('dark_correction.bil')
        
        dark_mean = np.mean(dark_image.image, axis=0)
        raw_minus_dark = self.image - dark_mean[None,:]

        return raw_minus_dark


    def get_spectralon_reflectance(self, wavelength):
        """
        Calculates the reflectance of the Spectralon panel for any
        given wavelength between 300 and 1500 nm. Note that the Pika L imager
        is limited to wavelengths between 387 and 1023 nm.

        :param wavelength: (float) wavelength in nanometers (3 sigfigs)
        :return: (float) reflectance of the Spectralon at that wavelength
        """

        step = 50 # 50 is the spectralon wavelength step

        lower = int(wavelength - (wavelength % step))
        upper = lower + step

        upperReflectance = self.spectralon_reflectance[upper]
        lowerReflectance = self.spectralon_reflectance[lower]

        slope = 1.0 * (upperReflectance - lowerReflectance) / step
        return slope * (wavelength - lower) + lowerReflectance


    ### Calculate the gain correction to apply to the rest of the spectra
    def DN_to_reflectance(self, image, cal, dark, coords):

        # Extract the mean and stddev of the dark correction image
        dark_mean = np.mean(dark, axis=0); dark_stddv = np.std(dark, axis=0);


        # Subtract dark correction from raw image
        ratminus_dark = image - dark_mean[None,:]


        # Extract the mean and stddev of the calibration image
        cal_mean = np.mean(cal, axis=0); cal_stddv = np.std(cal, axis=0);
        cal_mean = cal_mean - dark_mean[:,:640] # Subtract mean dark correction from mean calibration


        # Select Spectralon panel to calibrate DN to reflectances
        x1, x2, y1, y2  = coords
        panel = ratminus_dark[x1:x2,y1:y2,:] #data = dataCube[pointlist[:, 1], pointlist[:, 0], :]


        # Extract the mean and stddev of the panel image
        # panel_mean is the flat field to divide out
        panel_mean = np.mean(panel, axis=0); panel_stddv = np.std(panel, axis=0) #mean = np.mean(data, axis=0)


        # Extract Spectralon reflectances for each wavelength
        reflectances = np.array(map(get_spectralon_reflectance, self.imager_wavelengths[:240])).astype('float32')


        # Calculate DN -> reflectance correction for each wavelength
        correction = np.mean(reflectances[None,:] / panel_mean, axis=0)
        correction = correction[None,:][None,:] # Broadcast correction to 3d array


        return ratminus_dark * correction


    # Extract x,y coords from displayed plot (spectralon panel)
    # Instructions:
    #    Wait for the image to display
    #    Resize over image panel
    #    Close window
    #    The changed axes get stored and used as the Spectralon panel region to use as calibration
    def get_img_coords(self, hs_image, imgtitle):

        ### Insert the three wavelengths you want to make an 'rgb' image from the hyperspectral cube
        # The current list gives red, green and blue wavelengths (in nm) in that order
        rgb = [641, 551, 460]
        rgb_idxs = np.array([min(enumerate(self.imager_wavelengths), key=lambda x: abs(x[1] - wave))[0] for wave in rgb])

        red = hs_image[:,:,rgb_idxs[0]] / np.amax(hs_image[:,:,rgb_idxs[0]])
        green = hs_image[:,:,rgb_idxs[1]] / np.amax(hs_image[:,:,rgb_idxs[1]])
        blue = hs_image[:,:,rgb_idxs[2]] / np.amax(hs_image[:,:,rgb_idxs[2]])

        X, Y, S = hs_image.shape

        rgbArray = np.zeros((Y,X,3), 'uint8')

        rgbArray[..., 0] = red.T * 256
        rgbArray[..., 1] = green.T * 256
        rgbArray[..., 2] = blue.T * 256

        rgb_img = Image.fromarray(rgbArray)

        plt.figure()
        plt.imshow(rgb_img)#, origin="lower")
        plt.title(imgtitle, fontsize=20)
        axes = plt.gca()

        plt.show()

        # Not even deeper python magic, but it still took me forever to figure out
        x1, x2 = axes.get_xlim()
        y2, y1 = axes.get_ylim() # Not in order you would expect to account for imager rotation

        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        print "Panel coordinates: ", x1, x2, y1, y2

        return (x1, x2, y1, y2)

    def select_calibration_panel(self, imgtitle):
        """
        Allows user to draw a box around the calibration panel,
        and returns the coordinates of the box.
        """

        rgb = [641, 551, 460]
        rgb_idxs = np.array([min(enumerate(self.imager_wavelengths), key=lambda x: abs(x[1] - wave))[0] for wave in rgb])

        red = self.image[:,:,rgb_idxs[0]] / np.amax(self.image[:,:,rgb_idxs[0]])
        green = self.image[:,:,rgb_idxs[1]] / np.amax(self.image[:,:,rgb_idxs[1]])
        blue = self.image[:,:,rgb_idxs[2]] / np.amax(self.image[:,:,rgb_idxs[2]])

        X, Y, S = self.image.shape

        rgbArray = np.zeros((Y,X,3), 'uint8')

        rgbArray[..., 0] = red.T * 256
        rgbArray[..., 1] = green.T * 256
        rgbArray[..., 2] = blue.T * 256

        rgb_img = Image.fromarray(rgbArray)

        plt.figure()
        plt.imshow(rgb_img)#, origin="lower")
        plt.title(imgtitle, fontsize=20)
        axes = plt.gca()

        rs = RectangleSelector(axes, self.calibrate,
                       drawtype='box', useblit=False, button=[1], 
                       minspanx=5, minspany=5, spancoords='pixels', 
                       interactive=True)

        plt.show()


    ### Given a list of tuples which contain coordinates on a plant
    # extract the spectra from each of those coordinates
    def get_spectra(self, plant):

        spectra = []

        for loc in plant:

            x1, x2, y1, y2 = loc

            X = np.arange(x1,x2)
            Y = np.arange(y1,y2)

            plant_loc = np.array([ [xx,yy] for yy in Y for xx in X ])

            for coords in plant_loc:
                spectra.append(cal_images[0][coords[0], coords[1]])

        spectra = np.array(spectra)

        return spectra

    def display_rgb(self, imgtitle):

        ### Insert the three wavelengths you want to make an 'rgb' image from the hyperspectral cube
        # The current list gives red, green and blue wavelengths (in nm) in that order
        rgb = [641, 551, 460]
        rgb_idxs = np.array([min(enumerate(self.imager_wavelengths), key=lambda x: abs(x[1] - wave))[0] for wave in rgb])

        red = self.image[:,:,rgb_idxs[0]] / np.amax(self.image[:,:,rgb_idxs[0]])
        green = self.image[:,:,rgb_idxs[1]] / np.amax(self.image[:,:,rgb_idxs[1]])
        blue = self.image[:,:,rgb_idxs[2]] / np.amax(self.image[:,:,rgb_idxs[2]])

        X, Y, S = self.image.shape

        rgbArray = np.zeros((Y,X,3), 'uint8')

        rgbArray[..., 0] = red.T * 256
        rgbArray[..., 1] = green.T * 256
        rgbArray[..., 2] = blue.T * 256

        rgb_img = Image.fromarray(rgbArray)

        plt.figure()
        plt.imshow(rgb_img)#, origin="lower")
        plt.title(imgtitle, fontsize=20)
        axes = plt.gca()

        plt.show()