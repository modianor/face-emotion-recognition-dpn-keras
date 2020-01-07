# *_*coding:utf-8 *_*
#
#

import warnings

from keras import Sequential
from keras.engine.saving import model_from_json

warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

x = np.load('data/sfew_images.npy')
y = np.load('data/sfew_labels.npy')
# 从json文件中加载模型
with open('model/fer2013/model.json', 'r') as file:
    model_json = file.read()

model: Sequential = model_from_json(model_json)
model.load_weights('model/fer2013/model.h5')
y_pre = model.predict(x)
print(classification_report(np.argmax(y, axis=1), np.argmax(y_pre, axis=1)))
print(confusion_matrix(np.argmax(y, axis=1), np.argmax(y_pre, axis=1)))

print('\n精度: {:.4f}'.format(accuracy_score(np.argmax(y, axis=1), np.argmax(y_pre, axis=1))))
