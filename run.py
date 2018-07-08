import SparseAutoEncoder
import utils
import image
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

file_path = 'Banana_1_Day1.bil'
raw_image = image.HyperCube(file_path)
#raw_image.fix_image()

#raw_image.display_rgb("No Noise")

#noisy_image = utils.add_gaussian_noise(raw_image, stdev = 10)
#raw_image.display_rgb("Gaussian Noise")

original_shape = raw_image.image.shape

img_pieces = utils.divide_image(raw_image.image, size=10)
piece_shape = img_pieces[0].shape

noisy_imgs = utils.noisy_images(img_pieces)

flat_pieces = utils.flatten_multiple_images(noisy_imgs)

unflattened = utils.unflatten_multiple_images(flat_pieces, piece_shape)

recon = utils.reassemble_image(unflattened, original_shape)

raw_image.image = recon
#raw_image.display_rgb("Reconstructed")

################################################

n_inputs = flat_pieces[0].size
n_hidden = 100
n_iters = 1000


sae = SparseAutoEncoder.FeedforwardSparseAutoEncoder(n_inputs,n_hidden)
sae.training(flat_pieces[:100],n_iter=n_iters)

print "Running input through autoencoder..."
encoded = sae.encode(flat_pieces.astype('float32'))
decoded = sae.decode(encoded)
print "Done."


decoded_array = decoded.eval(session=sae.sess)

unflattened = utils.unflatten_multiple_images(decoded_array, piece_shape)
recon = utils.reassemble_image(unflattened, original_shape)
raw_image.image = recon
raw_image.display_rgb("Reconstructed")

# After training the model, an image of the representations (W1) will be saved
# Please check trained4000.png for example
#images=sae.W1.eval(sae.sess)
#print(images.shape)
#images=images.transpose()
#print(images.shape)

#visualizeW1(images,28,10,n_iters)