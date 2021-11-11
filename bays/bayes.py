import numpy as np


class NaiveBayesClassifier(object):
    def __init__(self):
        '''
        self.label_prob表示每种类别在数据中出现的概率
        例如，{0:0.333, 1:0.667}表示数据中类别0出现的概率为0.333，类别1的概率为0.667
        '''
        self.label_prob = {}
        '''
        self.condition_prob表示每种类别确定的条件下各个特征出现的概率
        例如训练数据集中的特征为 [[2, 1, 1],
                              [1, 2, 2],
                              [2, 2, 2],
                              [2, 1, 2],
                              [1, 2, 3]]
        标签为[1, 0, 1, 0, 1]
        那么当标签为0时第0列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第1列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第2列的值为1的概率为0，值为2的概率为1，值为3的概率为0;
        当标签为1时第0列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第1列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第2列的值为1的概率为0.333，值为2的概率为0.333,值为3的概率为0.333;
        因此self.condition_prob的值如下：     
        {
            0:{
                0:{
                    1:0.5
                    2:0.5
                }
                1:{
                    1:0.5
                    2:0.5
                }
                2:{
                    1:0
                    2:1
                    3:0
                }
            }
            1:
            {
                0:{
                    1:0.333
                    2:0.666
                }
                1:{
                    1:0.333
                    2:0.666
                }
                2:{
                    1:0.333
                    2:0.333
                    3:0.333
                }
            }
        }
        '''
        self.condition_prob = {}

    def fit(self, feature, label):
        '''
        对模型进行训练，需要将各种概率分别保存在self.label_prob和self.condition_prob中
        :param feature: 训练数据集所有特征组成的ndarray
        :param label:训练数据集中所有标签组成的ndarray
        :return: 无返回
        '''

        # ********* Begin *********#
        numfeature = len(feature)
        self.condition_prob = {0: {0: {1: 0.0, 2: 0.0}, 1: {1: 0.0, 2: 0.0}, 2: {1: 0.0, 2: 0.0, 3: 0.0}},
                               1: {0: {1: 0.0, 2: 0.0}, 1: {1: 0.0, 2: 0.0}, 2: {1: 0.0, 2: 0.0, 3: 0.0}}}
        self.label_prob = {0: 0.0, 1: 0.1}
        for i in range(numfeature):
            if (label[i] == 1):
                self.label_prob[0] = self.label_prob[0] + 1
                p1num += feature[i]
                for j in range(3):
                    self.condition_prob[1][j][feature[i][j]] += 1
            else:
                self.label_prob[1] += 1
        for j in range(3):
            self.condition_prob[0][j][1] /= self.label_prob[0]
            self.condition_prob[0][j][2] /= self.label_prob[0]
            self.condition_prob[1][j][1] /= self.label_prob[1]
            self.condition_prob[1][j][2] /= self.label_prob[1]
        self.condition_prob[0][2][3] /= self.label_prob[0]
        self.condition_prob[1][2][3] /= self.label_prob[1]
        self.label_prob[0] /= numfeature
        self.label_prob[1] /= numfeature

        # ********* End *********#

    def predict(self, feature):
        '''
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的ndarray
        :return:
        '''
        # ********* Begin *********#
        predicts = []
        for i in range(len(feature)):
            pgood = self.label_prob[1]  # good possibility
            pbad = self.label_prob[0]  # bad possibility
            for j in range(3):
                pgood *= self.condition_prob[1][j][feature[i][j]]
                pbad *= self.condition_prob[0][j][feature[i][j]]
            if pgood >= pbad:
                predicts.append(1)
            else:
                predicts.append(0)
        return predicts
        # ********* End *********#
