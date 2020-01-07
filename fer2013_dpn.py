# *_*coding:utf-8 *_*
#
import warnings

import numpy as np
from keras import callbacks
from keras.callbacks import CSVLogger

from dual_path_network import DualPathNetwork

warnings.filterwarnings("ignore")

X = np.load('data/data_images.npy')
Y = np.load('data/data_labels.npy')
X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
x_train, x_test = X[:20, :, :, :], X[20:40, :, :, :]
y_train, y_test = Y[:20, :], Y[20:40, :]

print(X.shape)

model = DualPathNetwork(input_shape=(112, 112, 1),
                        initial_conv_filters=8,
                        # initial_conv_filters=64,
                        depth=[3, 4, 10, 3],
                        # depth=[3, 4, 20, 3],
                        filter_increment=[2, 4, 8, 16],
                        # filter_increment=[16, 32, 24, 128],
                        cardinality=32,
                        width=3,
                        weight_decay=0,
                        include_top=True,
                        weights=None,
                        input_tensor=None,
                        pooling=None,
                        classes=7)

# print(model.summary())
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

checkpointer = callbacks.ModelCheckpoint(filepath="model/fer2013/checkpoint-{epoch:02d}.hdf5", verbose=1,
                                         save_best_only=True, monitor='val_acc', mode='max')
csv_logger = CSVLogger('model/fer2013/cnntrainanalysis3.csv', separator=',', append=False)
model.fit(x_train, y_train, nb_epoch=1, validation_data=(x_test, y_test),
          batch_size=100)  # ,callbacks=[checkpointer, csv_logger]
# model.save("model/fer2013/cnn_model.hdf5")

model_json = model.to_json()

with open('model/fer2013/model.json', 'w') as file:
    file.write(model_json)
model.save_weights('model/fer2013/model.h5')
