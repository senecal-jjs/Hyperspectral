import tensorflow as tf  
from sklearn.model_selection import train_test_split
import numpy as np 
import pickle 
import utils

tf.logging.set_verbosity(tf.logging.INFO)

def model(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 290])
    out = tf.layers.dense(input_layer, units=32, activation=tf.nn.relu)
    out = tf.layers.dense(out, units=32, activation=tf.nn.relu)
    out = tf.layers.dense(out, units=1, activation=None)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=out)

    # Calculate loss for both train and eval modes
    loss = tf.losses.mean_squared_error(predictions=out, labels=labels)

    # Configure the training op (for train mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for eval mode)
    eval_metric_ops = {"mean_absolute_error": tf.metrics.mean_absolute_error(labels=labels, predictions=out)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def preprocess(data_file, calc_features=False):
    try:
        produce_spectra = pickle.load( open( data_file, "rb" ))
    except (OSError, IOError) as e:
        print("Error loading pickled spectra!")
        produce_spectra = []

    reflectances = [item[:-1] for item in produce_spectra]
    baseline = reflectances[0]
    labels = [int(item[-1]) for item in produce_spectra]

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


def run(data_file):
    data_dict = preprocess(data_file, calc_features=False)

    # Create training and test data
    X_train, X_test, y_train, y_test = train_test_split(data_dict['feature'], data_dict['label'], test_size=0.33)
    y_train = np.reshape(np.asarray(y_train, dtype=np.float64), (len(y_train),1))
    y_test = np.reshape(np.asarray(y_test, dtype=np.float64), (len(y_test),1))

    # Creat the estimator
    regression_model = tf.estimator.Estimator(model_fn=model, model_dir="/tmp/test")

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.asarray(X_train)},
        y=y_train,
        batch_size=10,
        num_epochs=None,
        shuffle=True)

    regression_model.train(
        input_fn=train_input_fn,
        steps=10000)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.asarray(X_test)},
        y=y_test,
        num_epochs=1,
        shuffle=False)

    eval_results = regression_model.evaluate(input_fn=eval_input_fn)
    print(eval_results)

   
if __name__ == "__main__":
    run("Formatted_Data/banana.p")
