import numpy as np 
from sklearn.manifold import TSNE
import pickle 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import utils 


# Load banana spectra
try:
    produce_spectra = pickle.load( open( "Formatted_Data/potato1.p", "rb" ))
except (OSError, IOError) as e:
    produce_spectra = []

reflectances = [item[:-1] for item in produce_spectra]
labels = [int(item[-1]) for item in produce_spectra]


# plt.plot(reflectances[0], label=1)
# plt.plot(reflectances[5], label=2)
# plt.plot(reflectances[10], label=3)
# plt.show()
# plt.plot(reflectances[15], label=6)
# plt.plot(reflectances[20], label=7)
# plt.plot(reflectances[25], label=8)
# plt.plot(reflectances[30], label=9)
# plt.plot(reflectances[35], label=10)
# plt.legend(loc='upper left')
# print(labels)

reflectances = np.vstack(reflectances)
print(reflectances.shape)
X_embedded = TSNE(n_components=2, n_iter=5000, perplexity=10, learning_rate=100.0, metric=utils.spectral_info_divergence).fit_transform(reflectances)

# Assign colors 
# tomato
#colors = {1: 'red', 2: 'red', 3: 'red', 4:'red', 5: 'greeen', 6: 'green', 7: 'green', 8: 'green', 9: 'black', 10: 'black', 14: 'black'}
# potato
colors = {1: 'red', 2: 'red', 3: 'red', 4: 'red', 5: 'green', 6: 'green', 7: 'green', 8: 'black', 9: 'black', 10: 'black', 14: 'magenta', 16: 'magenta', 17: 'magenta', 20: 'magenta'}
# banana
#colors = {1: 'red', 2: 'red', 3: 'red', 4: 'red', 5: 'green', 6: 'green', 7: 'green', 8:'magenta', 9: 'magenta', 10: 'magenta', 13: 'magenta', 15: 'magenta'}
cmap = []
for label in labels:
    cmap.append(colors[label])

fig = plt.figure()
plt.scatter(X_embedded[:,0], X_embedded[:,1], color=cmap)# c=np.linspace(0, len(labels), len(labels)), cmap='winter')
# ax = Axes3D(fig)
# ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], color=cmap)
plt.show()