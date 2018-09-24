# -*- coding: UTF-8 -*-

import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from keras.utils import np_utils
from keras.models import load_model
from sklearn import preprocessing
import warnings

# 共169个特征
feature_size = 169
# 共4类
num_classes = 4

warnings.filterwarnings('ignore')

# 说明: 该文件为加载已有的模型（LSTM.h5或GRU.h5） 观测分类结果


def data_prepare():

    print("========Start Data_Prepare========")
    dataset1 = np.loadtxt('/home/whsse/Data/Test0408/公交车.txt', dtype=float)
    random.shuffle(dataset1)
    print dataset1.shape

    dataset2 = np.loadtxt('/home/whsse/Data/Test0408/小汽车.txt', dtype=float)
    random.shuffle(dataset2)
    print dataset2.shape

    dataset3 = np.loadtxt('/home/whsse/Data/Test0408/火车.txt', dtype=float)
    random.shuffle(dataset3)
    print dataset3.shape

    dataset4 = np.loadtxt('/home/whsse/Data/Test0408/地铁.txt', dtype=float)
    random.shuffle(dataset4)
    print dataset4.shape

    print("========Start Normalization========")
    # 标准化 z-score
    scaler1 = preprocessing.StandardScaler().fit(dataset1)
    scaler2 = preprocessing.StandardScaler().fit(dataset2)
    scaler3 = preprocessing.StandardScaler().fit(dataset3)
    scaler4 = preprocessing.StandardScaler().fit(dataset4)

    test_set1 = scaler1.transform(dataset1)
    test_set2 = scaler2.transform(dataset2)
    test_set3 = scaler3.transform(dataset3)
    test_set4 = scaler4.transform(dataset4)

    # 测试集贴标签
    label1 = 1 * np.ones((len(test_set1), 1))
    test_set1 = np.concatenate((test_set1, label1), axis=1)

    label2 = 2 * np.ones((len(test_set2), 1))
    test_set2 = np.concatenate((test_set2, label2), axis=1)

    label3 = 3 * np.ones((len(test_set3), 1))
    test_set3 = np.concatenate((test_set3, label3), axis=1)

    label4 = 4 * np.ones((len(test_set4), 1))
    test_set4 = np.concatenate((test_set4, label4), axis=1)

    test_data = np.concatenate((test_set1, test_set2, test_set3, test_set4), axis=0)

    print test_data.shape
    return test_data


def load_data(test_data):

    X_test = test_data[:, 0:feature_size]
    Y_test = test_data[:, feature_size]

    # 根据时间步进行的预处理操作
    time_step = 5

    X_test_list = X_test.tolist()
    temp_test = list()
    for i in range(0, len(X_test) - time_step):
        temp_test.extend(X_test_list[i: i + time_step])
    X_test_lstm = np.array(temp_test)
    X_test_lstm = X_test_lstm.reshape(len(X_test) - time_step, feature_size * time_step)
    # 更改样本形状 (m_sample, 5, 169)
    x_test = X_test_lstm.reshape(X_test_lstm.shape[0], 5, 169)
    x_test = x_test.astype('float32')

    Y_testx = Y_test[time_step: len(Y_test)]

    # 类别标签编码
    encoder = LabelEncoder()
    encoded_testY = encoder.fit_transform(Y_testx)
    y_test = np_utils.to_categorical(encoded_testY, num_classes)

    return x_test, y_test, Y_testx


def predict(model, x_test, y_test):
    predictions = model.predict(x_test)
    get_class = lambda classes_probabilities: np.argmax(classes_probabilities) + 1
    y_pred = np.array(map(get_class, predictions))
    if y_test is not None:
        y_true = np.array(map(get_class, y_test))
        print "准确率：" + str(accuracy_score(y_true, y_pred))
    return y_true, y_pred


# 自定义混淆矩阵打印
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def run_model(test_data):

    x_test, y_test, Y_testx = load_data(test_data)

    model = load_model('LSTM.h5')

    loss, accuracy = model.evaluate(x_test, y_test)

    print('test loss:', loss)
    print('accuracy:', accuracy)

    y_true, y_pred = predict(model, x_test, y_test)

    print(type(y_true))
    print(type(y_pred))
    print(y_true.shape)
    print(y_pred.shape)

    target_name = ['bus', 'car', 'train', 'metro']
    report = classification_report(y_true, y_pred, target_names=target_name)
    print(report)

    # 打印混淆矩阵
    cnf_matrix = confusion_matrix(Y_testx, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    class_names = ['bus', 'car', 'train', 'metro']
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
    plt.show()


if __name__ == '__main__':

    new_data = data_prepare()
    run_model(new_data)
