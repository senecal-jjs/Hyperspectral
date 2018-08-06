from utils import *
from keras_CNN import *
from keras_FFNN import *
import numpy as np


#pickle_a_day("tomato_small_cnn_6", "../YukonGold_Tomato_Banana_1_Day7.bil", "fresh", 1)

# [31  66  80  91 128 136 162 181 216 266]
wavelengths = [87,137,250,275]
produce_spectra = readfiles("tomato_cnn_")

labels = np.asarray([x[-1] for x in produce_spectra if x[0] != []])
count = collections.Counter(labels)
print(count)

cnn = CNN(produce_spectra, wavelengths)
cnn.train_cnn(generator=False)

#ffnn=FFNN(produce_spectra, wavelengths)
#ffnn.train_ffnn()