from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def classification(train_feature, train_label, test_feature):
    '''
    对test_feature进行红酒分类
    :param train_feature: 训练集数据，类型为ndarray
    :param train_label: 训练集标签，类型为ndarray
    :param test_feature: 测试集数据，类型为ndarray
    :return: 测试集数据的分类结果
    '''

    #********* Begin *********#
    # 实例化StandardScaler对象
    scaler = StandardScaler()
    # 用train_feature的均值和标准差来进行标准化，并将结果保存到std_trainfeature
    std_trainfeature = scaler.fit_transform(train_feature)
    std_testfeature=scaler.fit_transform(test_feature)
    # 生成K近邻分类器
    clf = KNeighborsClassifier()
    # 训练分类器
    clf.fit(std_trainfeature, train_label)
    # 进行预测
    predict_result = clf.predict(std_testfeature)
    return predict_result
    #********* End **********#