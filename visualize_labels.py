import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

n=1

try:
    label_array = pickle.load( open("image_labels.p", "rb" ))

except (OSError, IOError) as e:
    print "nothing found by that name..."

#print label_array
#print label_array.shape

label_array = np.transpose(label_array, axes=(1,0,2))

colors = np.empty((label_array.shape[0]*n, label_array.shape[1]*n, 3))


"""for y in range(label_array.shape[1]):
	for x in range(label_array.shape[0]):
		if label_array[x][y] == 'tomato':
			colors[x][y] = np.array([255,0,0])
		elif label_array[x][y] == 'banana':
			colors[x][y] = np.array([255,255,0])
		elif label_array[x][y] == 'potato':
			colors[x][y] = np.array([0,255,0])
		elif label_array[x][y] == 'background':
			colors[x][y] = np.array([0,0,0])
		elif label_array[x][y] == 'spectralon':
			colors[x][y] = np.array([255,255,255])"""


for y in range(colors.shape[1]):
	for x in range(colors.shape[0]):
		if label_array[int(x/n)][int(y/n)] == 'tomato':
			colors[x][y] = np.array([255,0,0])
		elif label_array[int(x/n)][int(y/n)] == 'banana':
			colors[x][y] = np.array([255,255,0])
		elif label_array[int(x/n)][int(y/n)] == 'potato':
			colors[x][y] = np.array([0,255,0])
		elif label_array[int(x/n)][int(y/n)] == 'background':
			colors[x][y] = np.array([0,0,0])
		elif label_array[int(x/n)][int(y/n)] == 'spectralon':
			colors[x][y] = np.array([255,255,255])

plt.title("Labeled with n=1", fontsize=20)
plt.imshow(colors)
plt.show()

"""overlay = Image.open("../../figures/all_produce_original.png")
background = Image.open("../../figures/labeled_n=10.png")

#background = background.convert("RGBA")
#overlay = overlay.convert("RGBA")

background.paste(overlay, (0, 0), overlay)
background.show()"""