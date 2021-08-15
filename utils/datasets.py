import os
import numpy as np

from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array, load_img

emotion_table = {'neutral'  : 0, 
                 'happiness': 1, 
                 'surprise' : 2, 
                 'sadness'  : 3, 
                 'anger'    : 4, 
                 'disgust'  : 5, 
                 'fear'     : 6, 
                 'contempt' : 7}

class Fer2013Plus(object):
    def __init__(self, folder="./datasets/fer2013plus", image_size=(48, 48)):
        """
        构造函数
        """
        self.folder = folder
        self.target_size = image_size

    def gen_train(self):
        """
        产生训练数据
        :return expressions:读取文件的顺序即标签的下标对应
        :return x_train: 训练数据集
        :return y_train： 训练标签
        """
        folder = os.path.join(self.folder, 'Training')
        # 这里原来是list出多个表情类别的文件夹，后来发现服务器linux顺序不一致，会造成问题，所以固定读取顺序
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=self.target_size, color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train

    def gen_valid(self):
        """
        产生验证集数据
        :return:
        """
        folder = os.path.join(self.folder, 'PublicTest')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_valid = []
        y_valid = []
        for i in tqdm(range(len(expressions))):
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=self.target_size, color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_valid.append(img)
                y_valid.append(i)
        x_valid = np.array(x_valid).astype('float32') / 255.
        y_valid = np.array(y_valid).astype('int')
        return expressions, x_valid, y_valid

    def gen_test(self):
        """
        产生验证集数据
        :return:
        """
        folder = os.path.join(self.folder, 'PrivateTest')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_test = []
        y_test = []
        for i in tqdm(range(len(expressions))):
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=self.target_size, color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_test.append(img)
                y_test.append(i)
        x_test = np.array(x_test).astype('float32') / 255.
        y_test = np.array(y_test).astype('int')
        return expressions, x_test, y_test