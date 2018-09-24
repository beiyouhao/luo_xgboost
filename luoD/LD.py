# -*- coding: UTF-8 -*-

import itertools
import matplotlib.pyplot as plt
import warnings
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.externals.joblib import dump, load

warnings.filterwarnings("ignore")

rng = np.random.RandomState(31337)

# np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

fea_size = 120


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


def main():
    print("--------  Start  --------")

    datax = np.load('/Users/wanghao/实验室/LuoD_Datax/new_clear_data_for_raw.npy')
    print(datax.shape)
    print(datax)
    datax = np.random.permutation(datax)

    print("train_test_split")
    x = datax[:, 0: fea_size]  # 总样本
    y = datax[:, fea_size]    # 总标签
    print(y)

    trains, vals = train_test_split(datax, test_size=0.3, random_state=1)

    # print("Step0: 网格搜索")
    # param_grid = {
    #     'max_depth': [3, 6, 9],
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'subsample': [0.5, 0.8, 1],
    #     'n_estimators': [100, 300, 500, 800],
    # }
    # clf = xgb.XGBClassifier(objective='multi:softmax', num_class=25)
    # gs = GridSearchCV(clf, param_grid, cv=10, n_jobs=4)
    # print 'begin to fit...'
    # gs.fit(x, y)
    # print gs.best_params_
    # print gs.cv_results_
    # dump(gs, 'xgb0915.model')

    print("Step1: 交叉验证")
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(x):
        xgb_model = xgb.XGBClassifier(max_depth=6, n_estimators=200, learning_rate=0.1, subsample=0.8,
                                      objective='multi:softmax', num_class=5) \
            .fit(x[train_index], y[train_index])
        print("test_acc:", np.mean(xgb_model.predict(x[test_index]) == y[test_index]))
        print("train_acc:", np.mean(xgb_model.predict(x[train_index]) == y[train_index]))
        print("-----------------------")

    # print("Step2: 训练测试7:3")
    # trains_x = trains[:, 0:fea_size]
    # trains_y = trains[:, fea_size]
    # print("train 70%: ", trains_x.shape, trains_y.shape)
    # trains_y = trains_y.tolist()
    # arr_appear = dict((a, trains_y.count(a)) for a in trains_y)
    # for ele in arr_appear:
    #     print(ele, arr_appear[ele])
    #
    # print("Begin training...")
    # xgb_model = xgb.XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.05, subsample=0.5,
    #                               objective='multi:softmax', num_class=25) \
    #     .fit(trains_x, trains_y)
    # print("Begin testing...")
    # vals_x = vals[:, 0:fea_size]
    # vals_y = vals[:, fea_size]
    # print("test 30%: ", vals_x.shape, vals_y.shape)
    # pred = xgb_model.predict(vals_x)
    # print(accuracy_score(vals_y, pred))
    # cnf_matrix = confusion_matrix(vals_y, pred)
    # class_names = range(25)
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
    # plt.show()

    # print("Step3: 树状图")
    # clf = xgb.XGBClassifier(max_depth=6, n_estimators=500, learning_rate=0.1, subsample=0.8)
    # xgb_model = clf.fit(x, y)
    # fig, ax = plt.subplots()
    # fig.set_size_inches(80, 60)
    # xgb.plot_tree(clf, ax=ax)
    # plt.show()
    # fig.savefig('xgb_tree.jpg')
    #
    # print("Step4: 评估特征重要性")
    # dtrain = xgb.DMatrix(x, label=y)
    # our_params = {'eta': 0.05, 'seed': 0, 'subsample': 0.5, 'min_child_weight': 1,
    #               'objective': 'multi:softmax', 'num_class': 25, 'max_depth': 3, }
    # num_round = 500
    # bst = xgb.train(our_params, dtrain, num_round)
    # importances = bst.get_fscore()
    # res = sorted(importances.items(), key=lambda importances: importances[1], reverse=True)
    # for i in res:
    #     print(i)

    print("--------  Finish  --------")


if __name__ == '__main__':
    main()
