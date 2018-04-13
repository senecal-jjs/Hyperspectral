import numpy as np
import pickle
from image import HyperCube
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.patches as mpatches
from sklearn.cross_decomposition import PLSRegression
from skimage import feature


''' This file contains utility functions that don't necessarily belong to the class
    hypercube, as they may not act on a single instantiation of the hypercube class '''

def spectral_correlation(spectra1, spectra2):
    '''
    Params: Two hypercubes from which the spectral correlation of two different
            regions will be compared.
    '''

    cube1_spectra = np.clip(np.array(spectra1), 0, None)
    cube2_spectra = np.clip(np.array(spectra2), 0, None)
    n = len(cube1_spectra)

    numerator = (n * np.sum(cube1_spectra * cube2_spectra)
                - (np.sum(cube1_spectra) * np.sum(cube2_spectra)))

    denominator = np.sqrt((n * np.sum(cube1_spectra**2) - np.sum(cube1_spectra)**2)
                        * (n * np.sum(cube2_spectra**2) - np.sum(cube2_spectra)**2))

    return numerator/denominator


def spectral_info_divergence(spectra1, spectra2):
    '''Ensure spectra our np arrays in order to apply the clip operation
       Clip operation is applied to avoid a divide by zero error in later
       operations'''
    cube1_spectra = np.clip(np.array(spectra1), 0.01, None)
    cube2_spectra = np.clip(np.array(spectra2), 0.01, None)

    cube1_probability = [wavelength/np.sum(cube1_spectra) for wavelength in cube1_spectra]
    cube2_probability = [wavelength/np.sum(cube2_spectra) for wavelength in cube2_spectra]

    n = len(cube1_probability)

    entropy_xy = np.sum([cube1_probability[i] * np.log2(cube2_probability[i]/cube1_probability[i])
                        for i in range(n)])

    entropy_yx = np.sum([cube2_probability[i] * np.log2(cube1_probability[i]/cube2_probability[i])
                        for i in range(n)])

    return np.abs(entropy_xy + entropy_yx)


def euclidean_distance(spectra1, spectra2):
    spectra1 = np.clip(np.array(spectra1), 0.0, None)
    spectra2 = np.clip(np.array(spectra2), 0.0, None)
    diff = np.abs(np.array(spectra1) - np.array(spectra2))
    return np.linalg.norm(diff)


def spectral_angle(spectra1, spectra2):
    spectra1 = np.clip(np.array(spectra1), 0.0, None)
    spectra2 = np.clip(np.array(spectra2), 0.0, None)
    numerator = np.sum(spectra1 * spectra2)
    denominator = np.sqrt(np.sum(spectra1**2)) * np.sqrt(np.sum(spectra2**2))
    return np.arccos(numerator/denominator)


def area(spectra, imager_wavelengths):
    spectra = np.clip(np.array(spectra), 0, None)
    return np.trapz(spectra, imager_wavelengths)


def create_spectra_list(image_names):
    """
    Given a list of image names/locations, return a
    list containing the average spectra of each image.
    """

    average_spectra = []

    for image in image_names:

        img = HyperCube(image)
        img.fix_image()
        img.select_region('Select Region', img.set_average_spectra)

        spectra = img.get_average_spectra()
        average_spectra.append(spectra)

    return average_spectra

def pickle_a_day(produce_type, image, age, num_regions=5):
    """
    Given an image name, and the age of the produce
    (in days) when the image was taken, calibrate the
    image for darkness and reflectance, select five
    regions from the image, find the average reflectances
    for each, pickling each average reflectance array as
    inputs and the age as the target.

    Specify the type of procude by spelling out the name
    of the produce in full, using lowercase letters, and
    underscores for spaces (if needed).
    """

    img = HyperCube(image)
    img.fix_image()

    try:
        produce_spectras = pickle.load( open( produce_type+".p", "rb" ))

    except (OSError, IOError) as e:
        produce_spectras = []

    for _ in range(num_regions):
        img.select_region('Select Region', img.set_average_spectra)

        spectra = np.append(img.get_average_spectra(), age) #add age as target
        produce_spectras.append(spectra)

    pickle.dump(produce_spectras, open( produce_type+".p", "wb" ) )

def clear_pickle(produce_type):
    """
    Clear the contents of a pickle file for a
    given type of produce.
    """

    pickle.dump([], open( produce_type+".p", "wb" ) )

def principal_components(data, n, standardize=False):
    """
    Return the top n principal components from the data.
    If standardize is set to True, standardize the
    """

    pca = PCA(n_components=n)
    pca.fit(data)

    if standardize:
        standardized_data = standardize_data(data)
        transformed_data = pca.transform(standardized_data)
    else:
        transformed_data = pca.transform(data)

    print("Explained variance: ", pca.explained_variance_ratio_)

    return transformed_data

def kernel_pca(data, n):
    """
    Return the top n principal components determined by
    the kernel PCA, using an RBF kernel
    """

    kpca = KernelPCA(kernel="cosine", n_components=n)
    transformed_data = kpca.fit_transform(data)

    #print kpca.alphas_

    return transformed_data

def plot_components(data, labels):
    """
    Given the primary components and labels, plot the
    primary components.
    """

    patches = []
    patches.append(mpatches.Patch(color='blue', label='1-3 days'))
    patches.append(mpatches.Patch(color='red', label='4-7 days'))
    patches.append(mpatches.Patch(color='green', label='8-10 days'))
    patches.append(mpatches.Patch(color='yellow', label='>10 days'))

    for i, data_point in enumerate(data):
        if labels[i] < 4:
            plt.scatter(data_point[0], data_point[1], color='b', marker='o')

        elif labels[i] > 3 and labels[i] < 8:
            plt.scatter(data_point[0], data_point[1], color='r', marker='+')

        elif labels[i] > 7 and labels[i] < 11:
            plt.scatter(data_point[0], data_point[1], color='g', marker='s')

        else:
            plt.scatter(data_point[0], data_point[1], color='y', marker='^')

    plt.legend(handles=patches)
    plt.title("Principal Components of Tomatoes\n"
        +"(all features, rbf)", fontsize=20)
    plt.xlabel("PC-1")
    plt.ylabel("PC-2")
    plt.show()

def pls_regression(data, labels, n=None):
    """
    Given the spectra and the age in days, run a
    partial least squares regression. Unless a
    number of components is specified, keep all
    components.
    """

    if not n:
        n = len(data[0])

    pls = PLSRegression(n_components=n)
    pls.fit(data, labels)

    params = pls.get_params()

    print(params)

def feature_selection(data, n):
    """
    Given a set of data, select the n most explanatory
    features using PCA.
    """

    pca = PCA(n_components=len(data[0]))
    #pca = PCA(n_components=n)
    pca.fit(data)

    norm_components = []
    for weight in np.absolute(pca.components_):
        norm = weight / np.sum(weight)
        norm_components.append(norm)

    weighted_sums = []
    for feature in np.transpose(norm_components):
        w_sum = np.dot(feature, pca.explained_variance_ratio_)
        weighted_sums.append(w_sum)

    print (weighted_sums)
    print ("-------------------------------")
    print (np.array(weighted_sums).argsort()[-n:][::-1])
    print (np.max(weighted_sums))

def kernel_feature_selection(data, n):
    """
    Given a set of data, select the n most explanatory
    features using kernel PCA.
    """

    #pca = PCA(n_components=len(data[0]))
    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, n_components=n)
    kpca.fit(data)

    norm_components = []
    for weight in np.absolute(kpca.components_):
        norm = weight / np.sum(weight)
        norm_components.append(norm)

    weighted_sums = []
    for feature in np.transpose(norm_components):
        w_sum = np.dot(feature, kpca.explained_variance_ratio_)
        weighted_sums.append(w_sum)

    print (weighted_sums)
    print ("-------------------------------")
    print (np.array(weighted_sums).argsort()[-n:][::-1])
    print (np.max(weighted_sums))

def pearson_coefficient_features(data, labels, n):
    """
    Given the input data and output labels, calculate
    the correlation criteria, used for ranking
    feature importance.
    """

    coefficients = []

    mean_centered_labels = np.subtract(labels, np.mean(labels))

    #Second term of denominator is the square root of the sum 
    #of the squared differences between label y_i and mean(y)
    denom_second_term = np.sqrt(np.dot(mean_centered_labels, mean_centered_labels))

    for i, row in enumerate(np.transpose(data)):

        mean_centered_row = np.subtract(row, np.mean(row))

        #Numerator is the dot product of the mean-centered row and label
        numerator = np.dot(mean_centered_row, mean_centered_labels)

        #First term in denominator is the square root of the sum
        #of the squared differences between data point x_i and mean(x)
        denom_first_term = np.sqrt(np.dot(mean_centered_row, mean_centered_row))

        pearson_coef = numerator/(denom_first_term*denom_second_term)
        coefficients.append(np.abs(pearson_coef))

    print (np.array(coefficients).argsort()[-n:][::-1])

def canny_edge_detection(image, title):
    """
    Given a grayscale image, detect the edges of the
    image using a canny filter
    """

    edges = feature.canny(image, sigma=10)


    # display results
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                                    sharex=True, sharey=True)

    ax1.imshow(image)#, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('binary image', fontsize=20)

    ax2.imshow(edges, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Canny filter, $\sigma=10$ - ' + title, fontsize=20)


    fig.tight_layout()

    plt.show()

def detect_produce(image):
    """
    Given an image, automatically detect the edges
    of the produce.
    """

    mean_image = np.mean(image.image, axis=2)
    mean_reflectance = np.mean(mean_image)
    stdev = np.std(mean_image)

    low = mean_reflectance + stdev
    high = low + (2*stdev)

    black_and_white = to_black_and_white(image, low, high)
    canny_edge_detection(black_and_white, "Produce")

def detect_spectralon(image):
    """
    Given an image, automatically detect the edges
    of the produce.
    """

    mean_image = np.mean(image.image, axis=2)
    mean_reflectance = np.mean(mean_image)
    stdev = np.std(mean_image)

    high = np.max(mean_image)
    low = high - (22.5*stdev)

    black_and_white = to_black_and_white(image, low, high)
    canny_edge_detection(black_and_white, "Spectralon")


def to_black_and_white(image, low, high):
    """
    Given an image (as a 3-d numpy array), take the
    mean of the spectra for each pixel to reduce to
    2-d. If the value is between low and high, set to
    one, else, set to zero
    """

    black_and_white = np.mean(image.image, axis=2)

    for i in range(len(black_and_white)):
        for j in range(len(black_and_white[i])):
            if black_and_white[i][j] >= low and black_and_white[i][j] <= high:
                black_and_white[i][j] = 1
            else:
                black_and_white[i][j] = 0

    return black_and_white

def add_gaussian_noise(image, stdev=10):
    """
    Given an image, add Gaussian noise
    for processing in a denoising 
    autoencoder
    """

    #Random noise with standard deviation of stdev
    noise = np.random.uniform(-stdev, stdev, image.image.shape)

    #Update the image with the generated noise
    return np.add(image.image, noise)

def divide_image(image, size=10):
    """
    Given a 3-d array representing an image,
    divide the image into 10*10 squares
    along the x and y axes, and return the
    hyperrectangles given by those squares. 
    Assumes width and height of the image 
    to be divisible by 10
    """

    hyperrectangles = []

    #Find the x and y dimensions of the image
    y_dim, x_dim = image.shape[0], image.shape[1]

    #Iterate left to right, top to bottom
    for y in range(size, y_dim+1, size):
        for x in range(size, x_dim+1, size):

            hyperrectangle = image[y-size:y, x-size:x,:]
            hyperrectangles.append(np.array(hyperrectangle))

    return np.array(hyperrectangles)

def reassemble_image(image_chunks, dimensions):
    """
    Given a segmented image and the original
    x,y dimensions of that image, piece the
    original image back together
    """

    reconstructed_image = []

    chunks_shape = image_chunks.shape
    num_chunks = chunks_shape[0]
    square_size = chunks_shape[1] #Width/height of each chunk
    num_rows = (num_chunks*square_size)/dimensions[1]
    row_size = (num_chunks*square_size)/dimensions[0]

    i = 0
    for _ in range(num_rows):
        row_chunk = []
        for __ in range(row_size):
            row_chunk.append(image_chunks[i])
            i+=1

        for k in range(square_size):
            row = []
            for j in range(row_size):
                row.extend(row_chunk[j][k])

            row = np.array(row)

            reconstructed_image.append(row)

    return np.array(reconstructed_image)

def flatten(img):
    """
    Given an image, flatten it to 1-d array
    """

    image_shape = img.shape
    flattened_dimension = image_shape[0] * image_shape[1] * image_shape[2]
    return np.reshape(img, flattened_dimension)

def unflatten(img, img_shape):
    """
    Given a flat image and a shape,
    reshape the image and return
    """

    return np.reshape(img, img_shape)

def flatten_multiple_images(img_list):
    """
    Given a list of images, return an
    array containing the flattened images
    """

    flat_imgs = []

    for img in img_list:
        flat = flatten(img)
        flat_imgs.append(flat)

    return np.array(flat_imgs)

def unflatten_multiple_images(img_list, img_shape):
    """
    Given a list of images and a shape
    return an array of reshaped images
    """

    unflattened_imgs = []

    for img in img_list:
        unflat = unflatten(img, img_shape)
        unflattened_imgs.append(unflat)

    return np.array(unflattened_imgs)

def add_noise(image, stdev=10):
    """
    Given an image, add Gaussian noise
    for processing in a denoising 
    autoencoder
    """

    #Random noise with standard deviation of stdev
    noise = np.random.uniform(-stdev, stdev, image.shape)

    #Update the image with the generated noise
    return np.add(image, noise)

def noisy_images(img_list, stdev=10):
    """
    Given a list of images, add noise
    to each image and return the
    noisy images
    """

    noisy_imgs = []

    for img in img_list:
        noisy_img = add_noise(img, stdev)
        noisy_imgs.append(noisy_img)

    return np.array(noisy_imgs)

def avg_spectra_divided_image(image, n=50, norm=True):
    """
    Split the image into n by n squares,
    then get the average spectra of 
    each square.
    """

    chunks = divide_image(image.image, n)

    avg_spectra = []
    for chunk in chunks:
        avg_spectra.append(np.mean(chunk, axis=(0,1)))

    avg_spectra = np.array(avg_spectra)
    
    if norm:
        avg_spectra = normalize(avg_spectra)

    return avg_spectra