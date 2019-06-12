import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection, metrics
import warnings

# 核心代码，设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 10000)
warnings.filterwarnings('ignore')

# 导入并合并数据
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
all_data = pd.concat([train_data, test_data], ignore_index=True)  # 合并数据
# print(train_data.columns.values.tolist())

# 查看数据的情况
# print(all_data.info())
# 查看某个特征缺失值个数
# print(train_data['Age'].isna().sum())

# 数据分析

# 存活情况
# train_data['Survived'].value_counts()

# sns.countplot(x="Sex", hue="Survived", data=train_data)
# 可以看出 female 的存活率高于 male

# sns.countplot(x="Pclass", hue="Survived", data=train_data)
# 可以看出 Pclass=1、2 的存活率高于 Pclass=3

# sns.countplot(x="SibSp", hue="Survived", data=train_data)
# 幸存率最高的是1和2

# 数据清洗
all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median())
# 归一化
all_data['Age'] = (all_data['Age'] - all_data['Age'].min()) / (
        all_data['Age'].max() - all_data['Age'].min())
all_data = all_data.drop(['Cabin'], axis=1)
all_data = all_data.drop(['PassengerId'], axis=1)
all_data['Embarked'] = all_data['Embarked'].fillna('S')
all_data['Fare'] = all_data['Fare'].fillna(26.55)
# 归一化
all_data['Fare'] = (all_data['Fare'] - all_data['Fare'].min()) / (
        all_data['Fare'].max() - all_data['Fare'].min())

all_data['Name'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
# 创建字典1
dict1 = {}
dict1.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))  # 这里的update是添加的意思，生成新dict添加到dict1
dict1.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
dict1.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
dict1.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
dict1.update(dict.fromkeys(['Mr'], 'Mr'))
dict1.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
# 根据dict1进行转换
all_data['Name'] = all_data['Name'].map(dict1)
# print(train_data)
# 根据字典2转换
all_data.loc[all_data['Name'] == 'Officer', 'Name'] = 0
all_data.loc[all_data['Name'] == 'Mr', 'Name'] = 0
all_data.loc[all_data['Name'] == 'Royalty', 'Name'] = 1
all_data.loc[all_data['Name'] == 'Mrs', 'Name'] = 1
all_data.loc[all_data['Name'] == 'Miss', 'Name'] = 1
all_data.loc[all_data['Name'] == 'Master', 'Name'] = 1
# print(all_data['Name'][0:5])

all_data.loc[all_data['Sex'] == 'female', 'Sex'] = 0
all_data.loc[all_data['Sex'] == 'male', 'Sex'] = 1

all_data.loc[all_data['Embarked'] == 'S', 'Embarked'] = 0
all_data.loc[all_data['Embarked'] == 'C', 'Embarked'] = 1
all_data.loc[all_data['Embarked'] == 'Q', 'Embarked'] = 2

ticket_count = dict(all_data['Ticket'].value_counts())
all_data['Ticket'] = all_data['Ticket'].map(ticket_count)


def setTicket(x):
    if 1 < x < 5:
        return 1
    else:
        return 0


all_data['Ticket'] = all_data['Ticket'].apply(lambda x: setTicket(x))

train_data = all_data[all_data['Survived'].notnull()]
test_data = all_data[all_data['Survived'].isnull()].drop('Survived', axis=1)

y = train_data['Survived']
X_train = train_data.drop('Survived', axis=1).as_matrix()
X_test = test_data.as_matrix()

pipe = Pipeline([('select', SelectKBest(k=8)),
                 ('classify', RandomForestClassifier(random_state=10, max_features='sqrt'))])
param_test = {'classify__n_estimators': list(range(20, 50, 2)),
              'classify__max_depth': list(range(3, 60, 3))}
gsearch = GridSearchCV(estimator=pipe, param_grid=param_test, scoring='roc_auc', cv=10)
gsearch.fit(X_train, y)
print(gsearch.best_params_, gsearch.best_score_)

select = SelectKBest(k=8)
clf = RandomForestClassifier(random_state=10, warm_start=True,
                             n_estimators=26,
                             max_depth=6,
                             max_features='sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(X_train, y)

cv_score = model_selection.cross_val_score(pipeline, X_train, y, cv=10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))

predictions = pipeline.predict(X_test)
print(predictions)
