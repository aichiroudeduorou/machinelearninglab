# encoding=utf8
import os
import pandas as pd
from sklearn.linear_model.perceptron import Perceptron
from sklearn.preprocessing import StandardScaler
import pandas as pd
if os.path.exists('./step2/result.csv'):
    os.remove('./step2/result.csv')

# ********* Begin *********#

# 获取训练数据
train_data = pd.read_csv('./step2/train_data.csv')
# 获取训练标签
train_label = pd.read_csv('./step2/train_label.csv')
train_label = train_label['target']
# 获取测试数据
test_data = pd.read_csv('./step2/test_data.csv')

# 标准化数据
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data=scaler.fit_transform(test_data)

clf = Perceptron()
clf.fit(train_data, train_label)
result = clf.predict(test_data)

df = pd.DataFrame(result, columns=["result"])
df.to_csv('./step2/result.csv')
# ********* End *********#



