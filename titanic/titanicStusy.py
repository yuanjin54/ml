import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
PassengerId = train_data['PassengerId']
all_data = pd.concat([train_data, test_data], ignore_index=True)
# print(train_data.head())

# 缺失值处理
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
# print(train_data.describe())

# 字符串转数值
# print(train_data['Sex'].unique())
train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 0
train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 1
# print(train_data['Embarked'].unique())
# sns.barplot(x="Embarked", y="Survived", data=train_data)
# plt.show()
train_data['Embarked'] = train_data['Embarked'].fillna('C')
train_data.loc[train_data['Embarked'] == 'S', 'Embarked'] = 0
train_data.loc[train_data['Embarked'] == 'C', 'Embarked'] = 1
train_data.loc[train_data['Embarked'] == 'Q', 'Embarked'] = 2

print(train_data['Cabin'].unique())
# sns.barplot(x="Cabin", y="Survived", data=train_data)
# plt.show()
# plt.figure(2)
# sns.barplot(x="Pclass", y="Survived", data=train_data)
# plt.show()
# plt.figure(3)
# sns.barplot(x="SibSp", y="Survived", data=train_data)
# plt.show()
# sns.barplot(x="Parch", y="Survived", data=train_data)
# plt.show()
