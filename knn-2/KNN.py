# encoding=utf8
import numpy as np
import operator

class kNNClassifier(object):
    def __init__(self, k):
        '''
        初始化函数
        :param k:kNN算法中的k
        '''
        self.k = k
        # 用来存放训练数据，类型为ndarray
        self.train_feature = None
        # 用来存放训练标签，类型为ndarray
        self.train_label = None

    def fit(self, feature, label):
        '''
        kNN算法的训练过程
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: 无返回
        '''

        # ********* Begin *********#
        self.train_feature = feature
        self.train_label = label
        # ********* End *********#

    def predict(self, feature):
        '''
        kNN算法的预测过程
        :param feature: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray或list
        '''

        # ********* Begin *********#
        result = []

        for feat in feature:
            diff = self.train_feature - feat
            sq_diff = diff ** 2
            dist = sq_diff.sum(axis=1) ** 0.5
            dist_index = dist.argsort()
            sorted_labels = self.train_label[dist_index]

            class_count = {}
            for i in range(self.k):
                label = sorted_labels[i]
                class_count[label] = class_count.get(label, 0) + 1
            sorted_class_count = sorted(
                class_count.items(), key=lambda x: x[1], reverse=True)
            result.append(sorted_class_count[0][0])
        # ********* End *********#
