# *_*coding:utf-8 *_*
#
#

import warnings

from keras import Sequential

warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from numpy import ndarray
from keras.models import model_from_json

x = np.load('data/ck_images.npy')

y = np.load('data/ck_labels.npy')

n_batch = round(x.shape[0] / 100)

y_pre_all = ''
y_all = ''
# 从json文件中加载模型
with open('model/fer2013/model.json', 'r') as file:
    model_json = file.read()

model: Sequential = model_from_json(model_json)
model.load_weights('model/fer2013/model.h5')

y_pre = model.predict(x)

for i in range(n_batch + 1):
    print("batch {}".format(i + 1))
    start = i * 100
    end = (i + 1) * 100
    if end >= x.shape[0]:
        x_ = x[start:, :, :, :]
        y_ = y[start:, :]
    else:
        x_ = x[start:end, :, :, :]
        y_ = y[start:end, :]

    print(x_.shape)
    y_pre = model.predict(x_)
    if not isinstance(y_pre_all, ndarray):
        y_pre_all = y_pre
        y_all = y_
    else:
        y_pre_all = np.vstack((y_pre_all, y_pre))
        y_all = np.vstack((y_all, y_))
print(classification_report(np.argmax(y_all, axis=1), np.argmax(y_pre_all, axis=1)))
print(confusion_matrix(np.argmax(y_all, axis=1), np.argmax(y_pre_all, axis=1)))
print(accuracy_score(np.argmax(y_all, axis=1), np.argmax(y_pre_all, axis=1)))
print('\n精度: {:.4f}'.format(accuracy_score(np.argmax(y_all, axis=1), np.argmax(y_pre_all, axis=1))))
