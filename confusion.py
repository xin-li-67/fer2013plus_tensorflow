import itertools
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

test_dir = './datasets/fer2013plus/PrivateTest'
input_shape = (48, 48, 1)
batch_size = 32

model = load_model("./emotion_models/ferplus_vgg")

test_generator = ImageDataGenerator().flow_from_directory(test_dir,
                                                          target_size=(48, 48),
                                                          color_mode='grayscale',
                                                          batch_size=32,
                                                          class_mode='categorical')

# labels = test_generator.class_indices
# labels = dict((v, k) for k, v in labels.items())
# print(labels)

labels = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']


test_faces, test_emotions = test_generator.__getitem__(123)

test_true = np.argmax(test_emotions, axis=1)
test_pred = np.argmax(model.predict(test_faces), axis=1)
print("Model Accuracy on test image is: {:.4f}".format(accuracy_score(test_true, test_pred)))

preds = model.predict(test_faces)
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

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

# plot normalized confusion matrix
plot_confusion_matrix(confusion, classes=labels, title='Normalized confusion matrix')
plt.show()