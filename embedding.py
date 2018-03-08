import numpy as np 
from sklearn.manifold import TSNE
import pickle 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import utils 


# Load banana spectra
try:
    produce_spectra = pickle.load( open( "banana.p", "rb" ))
except (OSError, IOError) as e:
    produce_spectra = []

reflectances = [item[:290] for item in produce_spectra]
labels = [int(item[290]) for item in produce_spectra]

# plt.plot(reflectances[0], label=1)
# plt.plot(reflectances[5], label=2)
# plt.plot(reflectances[10], label=3)
# plt.plot(reflectances[15], label=6)
# plt.plot(reflectances[20], label=7)
# plt.plot(reflectances[25], label=8)
# plt.plot(reflectances[30], label=9)
# plt.plot(reflectances[35], label=10)
# plt.legend(loc='upper left')
# print(labels)

reflectances = np.vstack(reflectances)
print(reflectances.shape)
X_embedded = TSNE(n_components=2, n_iter=5000, perplexity=5, learning_rate=100.0, metric=utils.spectral_info_divergence).fit_transform(reflectances)

fig = plt.figure()
plt.scatter(X_embedded[:,0], X_embedded[:,1])
#ax = Axes3D(fig)
#ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2])
plt.show()