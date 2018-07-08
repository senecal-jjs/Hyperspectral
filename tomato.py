from utils import *
from keras_CNN import *
import numpy as np

#pickle_a_day("tomato_cnn", "Tomato_1_Day9.bil", "old", 1)
produce_spectra = pickle.load( open( "tomato_cnn.p", "rb" ))
wavelengths = [144,145,146]

cnn = CNN(produce_spectra, wavelengths)
cnn.train_cnn()
