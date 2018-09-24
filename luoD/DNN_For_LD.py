# -*- coding: UTF-8 -*-

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, classification_report
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Masking
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from time import time

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

feature_size = 121


# 搭建DNN网络结构
def build_DNN_model():
    model = Sequential()

    model.add(Dense(512, activation='relu', input_dim=feature_size))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def ld_test():
    ld_data = np.load('new_all_wifi_for_fea81+norm.npy')
    # ld_data[np.isnan(ld_data)] = np.mean(ld_data[~np.nan(ld_data)])
    trains, vals = train_test_split(ld_data, test_size=0.1, random_state=1)

    x_train = trains[:, 0:feature_size]
    y_train = trains[:, feature_size]
    x_test = vals[:, 0:feature_size]
    y_test = vals[:, feature_size]

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # convert class vectors to binary class matrices
    encoder = LabelEncoder()
    encoded_trainY = encoder.fit_transform(y_train)
    encoded_testY = encoder.fit_transform(y_test)
    y_train = np_utils.to_categorical(encoded_trainY, 5)  # one hot encoding
    y_test = np_utils.to_categorical(encoded_testY, 5)

    model = build_DNN_model()
    model.summary()
    history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1,
                        validation_split=0.1)
    # summarize history for validation
    plt.figure()
    plt.grid(True, linewidth=0.4)
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['val_loss'], 'g--')
    plt.title('DNN: acc & loss')
    plt.ylabel('val_acc')
    plt.xlabel('Epoch')
    plt.legend(['acc', 'loss'], loc='center right')
    plt.show()

    scores = model.evaluate(x_test, y_test)
    print(scores)


def main():

    ld_test()


if __name__ == '__main__':
    main()
