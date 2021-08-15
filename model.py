from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPool2D
from tensorflow.keras.layers import Flatten, Dense, Activation

def VGG(input_shape=(48, 48, 1), num_classes=8):
    input_layer = Input(shape=input_shape)

    # block1
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_layer)
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)
    # block2
    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)
    # block3
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)
    # block4
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='last')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)
    # fc
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation(activation="relu")(x)
    x = Dropout(0.5)(x)
    # fc
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation(activation="relu")(x)
    x = Dropout(0.5)(x)
    
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model