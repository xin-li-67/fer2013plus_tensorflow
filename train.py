import os
import logging
import argparse
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

from model import *
from utils.visualize import plot_loss, plot_acc
from utils.datasets import Fer2013Plus

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--plot_history", type=bool, default=True)
opt = parser.parse_args()
logging.info("Parser: {}".format(opt))

base_path = './emotion_models/'
input_shape = (48, 48, 1)
num_classes = 8
his = None

expressions, x_train, y_train = Fer2013Plus(image_size=input_shape[:2]).gen_train()
_, x_valid, y_valid = Fer2013Plus(image_size=input_shape[:2]).gen_valid()
_, x_test, y_test = Fer2013Plus(image_size=input_shape[:2]).gen_test()
# target编码
y_train = to_categorical(y_train).reshape(y_train.shape[0], -1)
y_valid = to_categorical(y_valid).reshape(y_valid.shape[0], -1)
print("load FerPlus dataset successfully, it has {} train images and {} valid iamges".format(y_train.shape[0], y_valid.shape[0]))

model = VGG(input_shape, num_classes)
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

trained_models_path = base_path + 'ferplus'
trained_model_name = trained_models_path + '.h5'

callback = [
    # EarlyStopping(monitor='val_loss', patience=50),
    # ReduceLROnPlateau(monitor='lr', factor=0.1, patience=20),
    ModelCheckpoint(trained_model_name, monitor='val_accuracy', verbose=True, save_best_only=True, save_weights_only=True)
    ]

train_generator = ImageDataGenerator(rotation_range=10,
                                        width_shift_range=0.05,
                                        height_shift_range=0.05,
                                        horizontal_flip=True,
                                        shear_range=0.2,
                                        zoom_range=0.2).flow(x_train, y_train, batch_size=opt.batch_size)
valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)
history_ferplus = model.fit(train_generator,
                            steps_per_epoch=len(y_train)//opt.batch_size,
                            epochs=opt.epochs,
                            validation_data=valid_generator,
                            validation_steps=len(y_valid)//opt.batch_size,
                            callbacks=callback)
his = history_ferplus

# test
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
print("test accuacy", np.sum(pred.reshape(-1) == y_test.reshape(-1)) / y_test.shape[0])

# save the best performance ckp into a whole model
model.load_weights(trained_model_name)
model.save('./emotion_models/ferplus_vgg')

if opt.plot_history:
    plot_loss(his.history, opt.dataset)
    plot_acc(his.history, opt.dataset)