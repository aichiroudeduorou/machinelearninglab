# encoding=utf8
import numpy as np


# 构建感知机算法
class Perceptron(object):
    def __init__(self, learning_rate=0.01, max_iter=200):
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, data, label):
        '''
        input:data(ndarray):训练数据特征
              label(ndarray):训练数据标签
        output:w(ndarray):训练好的权重
               b(ndarry):训练好的偏置
        '''
        # 编写感知机训练方法，w为权重，b为偏置
        self.w = np.array([1.] * data.shape[1])
        self.b = np.array([1.])
        # ********* Begin *********#
        length = data.shape[0]  # 数据集长度

        for i in range(self.max_iter):  # max_iter:最大迭代次数
            flag_error = True  # error标记

            for i in range(length):  # 迭代训练数据
                x = data[i]  # x:当前数据
                y = x.dot(self.w) + self.b

                if y < 0:
                    predict_y = 1
                else:
                    predict_y = -1
                if predict_y == label[i]:  # 预测对了
                    continue
                else:  # 预测错了
                    flag_error = True
                    # 更新 w 和 b
                    self.w += self.lr * predict_y * x
                    self.b += self.lr * predict_y
            if flag_error:
                break
        # ********* End *********#

    def predict(self, data):
        '''
        input:data(ndarray):测试数据特征
        output:predict(ndarray):预测标签
        '''
        # ********* Begin *********#
        predict = []
        for x in data:
            y = self.w.dot(x) + self.b
            if y < 0:
                predict_y = 1
            else:
                predict_y = -1
            predict.append(predict_y)
        # ********* End *********#
        return predict
