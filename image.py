import gdal
import numpy as np
from PIL import Image


class Image():
    def __init__(self, raw_bil_file):
        self.image = convert_bil_to_array(raw_bil_file)


    def convert_bil_to_array(self, raw_bil_file):
        # Open the .bil file
        raw_image = gdal.open(raw_bil_file)

        # Read in each spectral band, (there are 240 bands)
        image = np.dstack([read_bil_file(raw_image, ii) for ii in range(1, 241)]).astype('float32')

        return image


    ### Read in .bil files as np array
    # rasterband is the above defined spectral_bands
    def read_bil_file(self, img, rasterband):
        gdal.GetDriverByName('EHdr').Register()
        band = img.GetRasterBand(rasterband)
        data = band.ReadAsArray()
        return data


    def calibrate(self, calibration_bil):
        calibration = np.dstack([readbilfile(calibration_bil, ii).astype('float32') for ii in range(1, 241)])
        return calibration


    def dark_correction(self, dark_correction_bil):
        dark_correction = np.dstack([readbilfile(dark_correction_bil,ii).astype('float32') for ii in range(1, 241)])
        return dark_correction


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

        upperReflectance = spectralon_reflectance[upper]
        lowerReflectance = spectralon_reflectance[lower]

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
        reflectances = np.array(map(getSpectralonReflectance, imager_wavelengths[:240])).astype('float32')


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
        rgb_idxs = np.array([min(enumerate(imager_wavelengths), key=lambda x: abs(x[1] - wave))[0] for wave in rgb])

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
