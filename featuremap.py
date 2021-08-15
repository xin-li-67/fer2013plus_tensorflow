import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

from model import *

def get_feature_map(model, layer_index, channels, input_img):
    layer = K.function([model.layers[0].input], [model.layers[layer_index].output])
    feature_map = layer([input_img])[0]
    plt.figure(figsize=(20, 8))

    for i in range(channels):
        img = feature_map[:, :, :, i]
        plt.subplot(4, 8, i + 1)
        plt.imshow(img[0], cmap='gray')
    plt.savefig('rst.png')
    plt.show()

def plot_feature_map():
    model = CNN3()
    model.load_weights('./emotion_models/fer2013plus/vgg_ferplus.h5')

    img = cv2.cvtColor(cv2.imread('./demo/happy.png'), cv2.cv2.COLOR_BGR2GRAY)
    img.shape = (48, 48, 1)
    get_feature_map(model, 4, 32, img)

if __name__ == '__main__':
    plot_feature_map()