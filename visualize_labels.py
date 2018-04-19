import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense
import keras 
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def assess_health(image_labels, refl_curves, model_paths, pixel_size=1):
	'''Parameters - label_array: The pixel classifications
					model_paths: dictionary containing filepath to each produce health model '''
	n=pixel_size
	label_array = image_labels
	reflectances = refl_curves

	# plt.plot(reflectances[0])
	# plt.show()

	print(label_array.shape)
	print(reflectances.shape)

	# Load health models
	models = {}
	for produce_type in model_paths.keys():
		name = str(produce_type)
		model=Sequential()
		model.add(Dense(64, activation="relu",input_dim=290))
		model.add(Dense(64, activation="relu"))
		model.add(Dense(1, activation='linear'))
		model.compile(optimizer='adam', loss='mse', metrics=['mae'])
		model.load_weights(model_paths[produce_type])
		models[name] = model

	colors = np.empty((label_array.shape[0], label_array.shape[1]))

	# track mininum and maximum predictions for normalization to a color gradient later 
	min_day = 100
	max_day = -1

	refl_index = 0
	for y in range(label_array.shape[1]):
		for x in range(label_array.shape[0]):
			produce_type = label_array[x][y][0]
			if produce_type == 'background':
				colors[x][y] = np.array([10])
			elif produce_type == 'spectralon':
				colors[x][y] = np.array([12])
			else:
				health = models[produce_type].predict(np.array([reflectances[x][y]]), batch_size=1)

				if refl_index % 100 == 0:
					print(produce_type)
					print(health)
				colors[x][y] = health
				refl_index += 1

				if produce_type == 'banana':
					plt.plot(reflectances[x][y])
					plt.show()
	# loop through one more time and set background and spectralon back to values you want 
	# Convert predictions to colormap
	# print("min color: {0}\n max color: {1}".format(min_day,max_day))
	# norm = Normalize(vmin=min_day,vmax=max_day)
	# cmap = plt.cm.winter
	# print(cmap(norm(colors[x][y][0])))

	# for y in range(label_array.shape[1]):
	# 	for x in range(label_array.shape[0]):
	# 		if colors[x][y].all() != np.array([0,0,0]).all() and colors[x][y].all() != np.array([255,255,255]).all():
	# 			colors[x][y] = cmap(norm(colors[x][y][0]))

	return colors


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return [r, g, b]


if __name__ == "__main__":
	refl = pickle.load(open("image_refl.p","rb"))
	labels = pickle.load(open("image_labels.p","rb"))
	model_paths = {'banana': 'Models/banana_net.h5', 'potato': 'Models/potato_net.h5', 'tomato': 'Models/tomato_net.h5'}
	colors = assess_health(labels, refl, model_paths, pixel_size=10)
	# plt.title("Labeled with n=1", fontsize=20)
	shape = colors.shape
	print("color shape: {0}".format(shape))
	plt.imshow(np.transpose(colors, (1,0)), cmap='nipy_spectral')
	plt.colorbar()
	plt.show()

	# """overlay = Image.open("../../figures/all_produce_original.png")
	# background = Image.open("../../figures/labeled_n=10.png")

	# #background = background.convert("RGBA")
	# #overlay = overlay.convert("RGBA")

	# background.paste(overlay, (0, 0), overlay)
	# background.show()"""
