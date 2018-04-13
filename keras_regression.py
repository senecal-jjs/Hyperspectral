from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import utils
import pickle
import os


def save_model(filename, model):
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


def run_model(datafile, model_name):
    # Create the MLP
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=290))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile model with optimizer and loss function
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Configure data
    data = preprocess(datafile)
    features = data['feature']
    labels = data['label']
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)

    scores = []

    for train, test in kfold.split(features, labels):
        model.fit(features[train], labels[train], epochs=50, batch_size=5, verbose=0)

        # evaluate the model
        score = model.evaluate(features[test], labels[test], verbose=0)
        print("{0}: {1}".format(model.metrics_names[1], score[1]))
        scores.append(score[1])

    save_model(model_name, model)
    return scores


if __name__ == "__main__":
    result = run_model("Formatted_Data/tomato1.p", "tomato_net")
    print("Standard Dev: {0}".format(np.std(result)))




