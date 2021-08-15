from model import VGG

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

K.clear_session()

test_dir = './datasets/fer2013plus/PrivateTest'

# Create a basic model instance
model = VGG()
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(48, 48),
                                                  color_mode='grayscale',
                                                  batch_size=32,
                                                  class_mode='categorical')

loss, acc = model.evaluate(test_generator, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Loads the weights
model.load_weights('./emotion_models/fer2013plus.h5')

# Re-evaluate the model
loss, acc = model.evaluate(test_generator, verbose=2)
print("Restored weights, accuracy: {:5.2f}%".format(100*acc))

# Loads the model
model = load_model('./emotion_models/ferplus_vgg')

# Re-evaluate the model
loss, acc = model.evaluate(test_generator, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))