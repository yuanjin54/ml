import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 核心代码，设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 10000)
warnings.filterwarnings('ignore')

# 读取训练数据和测试数据
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
# 查看columns，发现test_data少了Survived，因为待测试
print(train_data.columns.values.tolist())
print(test_data.columns.values.tolist())

all_data = pd.concat([train_data, test_data], ignore_index=True)
print("train data:", train_data.shape[0], "行,", train_data.shape[1], "列")
print("test data size:", test_data.shape[0], "行,", test_data.shape[1], "列")
print("all data size:", all_data.shape[0], "行,", all_data.shape[1], "列")
# 统计每个特征nan的个数
# print(all_data.isnull().sum())
# print(test_data.isnull().sum())
# print(all_data['Cabin'].isna().sum())
# 1309 行, 12 列

# 缺失值处理
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())  # 年龄均值填充
train_data['Embarked'] = train_data['Embarked'].fillna('C')
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())

print(train_data.isnull().sum())
print(test_data.isnull().sum())
train_data.loc[train_data['Fare'] > 300, 'Fare'] = 300
# plt.plot(train_data['Fare'])
# plt.show()
# 字符串转数值
# print(train_data['Sex'].unique())
train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 0
train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 1

# 画图可知Embarked特征的C类最多，因此NaN用C填充
# print(train_data['Embarked'].unique())
# sns.barplot(x="Embarked", y="Survived", data=train_data)
# plt.show()
train_data.loc[train_data['Embarked'] == 'S', 'Embarked'] = 0
train_data.loc[train_data['Embarked'] == 'C', 'Embarked'] = 1
train_data.loc[train_data['Embarked'] == 'Q', 'Embarked'] = 2

# print(train_data['Cabin'].unique())
# sns.barplot(x="Cabin", y="Survived", data=train_data)
# 暂时不考虑这一列
train_data = train_data.drop(['Cabin'], axis=1)
Ticket_Count = dict(all_data['Ticket'].value_counts())
train_data['Ticket'] = train_data['Ticket'].apply(lambda x: Ticket_Count[x])
# sns.barplot(x="Ticket", y="Survived", data=train_data)
# plt.show()

train_data['Name'] = train_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
# sns.barplot(x="Title", y="Survived", data=train_data)
# plt.show()
train_data.loc[train_data['Name'] == 'Officer', 'Name'] = 0
train_data.loc[train_data['Name'] == 'Mr', 'Name'] = 0
train_data.loc[train_data['Name'] == 'Royalty', 'Name'] = 1
train_data.loc[train_data['Name'] == 'Mrs', 'Name'] = 1
train_data.loc[train_data['Name'] == 'Miss', 'Name'] = 1
train_data.loc[train_data['Name'] == 'Master', 'Name'] = 1

# plt.figure(3)
# sns.barplot(x="SibSp", y="Survived", data=train_data)
# plt.show()
# sns.barplot(x="Parch", y="Survived", data=train_data)
# plt.show()

# 归一化Age和Fare
train_data['Age'] = (train_data['Age'] - train_data['Age'].min()) / (train_data['Age'].max() - train_data['Age'].min())
train_data['Fare'] = (train_data['Fare'] - train_data['Fare'].min()) / (
            train_data['Fare'].max() - train_data['Fare'].min())

print(train_data.head(10))

##### 添加模型开始训练

