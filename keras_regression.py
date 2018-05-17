from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import utils
import pickle
import os
from scipy import stats
import matplotlib.pyplot as plt 


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)


def save_model(filename, model):
    print("Saving model!")
    checkpoint_dir = os.path.join(os.getcwd(), "Models")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, filename)
    model.save_weights(checkpoint_path)


def preprocess(data_file, calc_features=False):
    try:
        produce_spectra = pickle.load( open( data_file, "rb" ))
    except (OSError, IOError) as e:
        print("Error loading pickled spectra!")
        produce_spectra = []

    reflectances = np.array([item[:-1] for item in produce_spectra])
    baseline = reflectances[0]
    labels = np.array([int(item[-1]) for item in produce_spectra])

    if calc_features:
        feature_vectors = []
        for curve in reflectances:
            div = utils.spectral_info_divergence(baseline, curve)
            corr = utils.spectral_correlation(baseline, curve)
            dist = utils.euclidean_distance(baseline, curve)
            angle = utils.spectral_angle(baseline, curve)
            feature_vectors.append([div,corr,dist,angle])

        return {'feature': np.vstack(feature_vectors), "label": labels}
    else:
        return {'feature': np.vstack(reflectances), "label": labels}


def run_model(datafile, model_name, cv=False):
    # Create the MLP
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=290))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile model with optimizer and loss function
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # save weights for reintialization at each fold 
    weights = model.get_weights()

    # Configure data
    data = preprocess(datafile)
    features = data['feature']
    labels = data['label']

    early_features = []
    early_labels = []

    # only look at early days
    for i in range(len(labels)):
        if labels[i] < 5:
            early_features.append(features[i,:])
            early_labels.append(labels[i])

    features = np.array(early_features)
    labels = np.array(early_labels)

    print("{0} data points!".format(len(labels)))

    if cv:
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
        scores = []
        
        for train, test in kfold.split(features, labels):
            shuffle_weights(model, weights=weights)
            model.fit(features[train], labels[train], epochs=70, batch_size=5, verbose=0)

            # evaluate the model
            score = model.evaluate(features[test], labels[test], verbose=0)
            print("{0}: {1}".format(model.metrics_names[1], score[1]))
            scores.append(score[1])
        
        return scores
    else: 
        model.fit(features, labels, epochs=50, batch_size=5, verbose=1)
        save_model(model_name, model)


if __name__ == "__main__":
    result = run_model("Formatted_Data/peppers.p", "peppers_net.h5", cv=True)
    print("Mean: {0}\n StDev: {1}".format(np.mean(result),np.std(result)))
    # k2, p = stats.normaltest(result) 
    # alpha = .05

    # if p < alpha:
    #     print("Data not normal")
    

    plt.hist(result, bins='auto')
    plt.show()




