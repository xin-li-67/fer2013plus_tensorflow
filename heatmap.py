import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

K.clear_session()
img_path = './demo/neutral.jpg'
out_path = './demo/neutral_hm.jpg'
input_shape = (48, 48, 1)
num_classes = 8
expressions = {0 : 'anger', 
               1 : 'disgust', 
               2 : 'fear', 
               3 : 'happy', 
               4 : 'sad', 
               5 : 'surprised', 
               6 : 'neutral', 
               7 : 'contempt'}

model_path = './emotion_models/ferplus_vgg'
model = load_model(model_path)

img = load_img(img_path, target_size=input_shape, color_mode="grayscale")
face = img_to_array(img)
face = np.array(face).astype('float32') / 255
face = np.expand_dims(face, axis=0)

preds = model.predict(face)
index = np.argmax(preds[0])
print("The Emotion is [{}] as predicted with model: {}".format(expressions[index], model_path))

last_conv_layer = model.get_layer(index=-12)
heatmap_model = models.Model([model.inputs], [last_conv_layer.output, model.output])

# Get gradient of the winner class w.r.t. the output of the (last) conv. layer
with tf.GradientTape() as gtape:
    conv_output, predictions = heatmap_model(face)
    loss = predictions[:, np.argmax(predictions[0])]
    grads = gtape.gradient(loss, conv_output) 
    pooled_grads = K.mean(grads, axis=(0,1,2))

heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
heatmap = np.maximum(heatmap, 0)
max_heat = np.max(heatmap)

if max_heat == 0:
    max_heat = 1e-10
heatmap /= max_heat
heatmap = np.squeeze(heatmap)

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 1 + img

cv2.imwrite(out_path, superimposed_img)