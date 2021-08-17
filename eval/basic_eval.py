# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import numpy as np
from model import VGG

from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.datasets import Fer2013Plus

K.clear_session()

test_dir = './datasets/fer2013plus/PrivateTest'

# # Create a basic model instance
# model = VGG()
# sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

_, x_test, y_test = Fer2013Plus(image_size=(48,48)).gen_test()
y_test = to_categorical(y_test).reshape(y_test.shape[0], -1)

# loss, acc = model.evaluate(x_test, y_test, verbose=0)
# print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Loads the model
model = load_model('./emotion_models/ferplus_vgg')

# Re-evaluate the model
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
