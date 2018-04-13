import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.cm as cm
from matplotlib.colors import Normalize


# Method to classify the produce in an image
def classify_new_image(self):
	"""
	Load the network, then classify each
	square of a divided image
	"""

	n=10
	file_path = 'YukonGold_Tomato_Banana_1_Day3.bil'
	print ("Loading image...")
	raw_image = image.HyperCube(file_path)
	raw_image.dark_correction()
	original_shape = raw_image.image.shape
	orig_x = original_shape[0]
	orig_y = original_shape[1]
	print ("Dividing image...")
	divided_image_reflectances = utils.avg_spectra_divided_image(raw_image, n)

	input_size = len(self.inputs[0])
	output_size = len(self.valid_labels)
	hidden_size_1 = 15
	hidden_size_2 = 15

	print ("Loading model...")
	# model = self.train_mlp()

	print ("Classifying image...")
	return (model.predict(divided_image_reflectances), divided_image_reflectances)


def assess_health(pixel_size=1, image, model_paths):
	'''Parameters - label_array: The pixel classifications
					model_paths: dictionary containing filepath to each produce health model '''
	n=pixel_size
	label_array = image[0]
	reflectances = image[1]

	# Load health models
	models = {}
	for produce_type in model_paths.keys():
		name = str(produce_type)
		model = Sequential()
		model.load_weights(model_paths[produce_type])
		models[name] = model

	# try:
	#     label_array = pickle.load( open("image_labels.p", "rb" ))

	# except (OSError, IOError) as e:
	#     print "nothing found by that name..."

	# #print label_array
	# #print label_array.shape

	label_array = np.transpose(label_array, axes=(1,0,2))
	colors = np.empty((label_array.shape[0]*n, label_array.shape[1]*n, 3))

	for y in range(label_array.shape[1]):
		for x in range(label_array.shape[0]):
			produce_type = label_array[x][y]
			if produce_type = 'background':
				colors[x][y] = np.array([0,0,0])
			elif produce_type = 'spectralon':
				colors[x][y] = np.array([255,255,255])
			else:
				health = models[produce_type].predict(reflectances[x][y])
				colors[x][y] = health

	# Convert predictions to colormap
	cmap = cm.winter
	min = np.min(colors)
	max = np.max(colors)
	norm = Normalize(vmin=min, vmax=max)
	colors = cmap(norm(colors))

	# for y in range(colors.shape[1]):
	# 	for x in range(colors.shape[0]):
	# 		if label_array[int(x/n)][int(y/n)] == 'tomato':
	# 			colors[x][y] = np.array([255,0,0])
	# 		elif label_array[int(x/n)][int(y/n)] == 'banana':
	# 			colors[x][y] = np.array([255,255,0])
	# 		elif label_array[int(x/n)][int(y/n)] == 'potato':
	# 			colors[x][y] = np.array([0,255,0])
	# 		elif label_array[int(x/n)][int(y/n)] == 'background':
	# 			colors[x][y] = np.array([0,0,0])
	# 		elif label_array[int(x/n)][int(y/n)] == 'spectralon':
	# 			colors[x][y] = np.array([255,255,255])

	# plt.title("Labeled with n=1", fontsize=20)
	# plt.imshow(colors)
	# plt.show()

	# """overlay = Image.open("../../figures/all_produce_original.png")
	# background = Image.open("../../figures/labeled_n=10.png")

	# #background = background.convert("RGBA")
	# #overlay = overlay.convert("RGBA")

	# background.paste(overlay, (0, 0), overlay)
	# background.show()"""