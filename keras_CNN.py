import keras
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.layers import *
from keras.callbacks import ModelCheckpoint
import numpy as np

class CNN:

    def __init__(self, data, wavelengths):
        self.data = np.asarray([np.ndarray.take(x[0],wavelengths, axis=2) for x in data])
        self.img_shape = self.data[0].shape
        self.data_shape = ((len(data),), self.img_shape)
        self.data_shape = sum(self.data_shape, ())
        self.labels = np.asarray([x[-1] for x in data])
        self.unique_labels = set(self.labels)

    def train_cnn(self):
        input_size = self.img_shape
        output_size = len(self.unique_labels)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=input_size))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(output_size, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        x_train = self.data
        x_test = self.data

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(self.labels)
        y_test = encoder.fit_transform(self.labels)
        print(y_train)

        y_train = keras.utils.to_categorical(y_train, output_size)
        y_test = keras.utils.to_categorical(y_test, output_size)

        filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        model.fit(x_train, y_train,
                  validation_split=0.33,
                  batch_size=25,
                  epochs=3,
                  verbose=2,
                  #validation_data=(x_test, y_test),
                  #callbacks=callbacks_list
                  )
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])



