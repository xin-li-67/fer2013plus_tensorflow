import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from utils.datasets import Fer2013Plus

test_dir = './datasets/fer2013plus/PrivateTest'
input_shape = (48, 48, 1)
batch_size = 32

model = load_model("./emotion_models/ferplus_vgg")

_, x_test, y_test = Fer2013Plus(image_size=(48,48)).gen_test()
y_test = to_categorical(y_test).reshape(y_test.shape[0], -1)

labels = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']

test_true = np.argmax(y_test, axis=1)
test_pred = np.argmax(model.predict(x_test), axis=1)
print("Model Accuracy on test image is: {:.4f}".format(accuracy_score(test_true, test_pred)))

preds = model.predict(x_test)
print(classification_report(test_true, test_pred, zero_division=0)) # Output classification report (accuracy rate, regression rate, F1)_ score）

classes = [x for x in range(len(labels))]
confusion = confusion_matrix(test_true, test_pred, labels=classes)
print("Confusion result: \n", confusion)

# list_diag = np.diag(confusion) 
# print("list_diag: ", list_diag)

# list_raw_sum = np.sum(confusion, axis=1)
# print("list_raw_sum: ", list_raw_sum)

# each_acc = np.nan_to_num(list_diag.astype('Float32')/list_raw_sum.astype('Float32'))
# print("The accuracy of each label is {}".format(each_acc))

# ave_acc = np.mean(each_acc)
# print("The average accuracy of test faces is {}".format(ave_acc))

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    plt.title(title)
    sns_hmp = sns.heatmap(cm, 
                          annot=True, 
                          xticklabels=[classes[i] for i in range(len(classes))], 
                          yticklabels=[classes[i] for i in range(len(classes))], 
                          fmt=".2f")
    fig = sns_hmp.get_figure()
    fig.savefig('./heatmap.jpg', dpi=250)

# plot normalized confusion matrix
plot_confusion_matrix(confusion, classes=labels, title='Normalized Confusion Matrix')
plt.show()