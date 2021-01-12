import numpy as np
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

data_path = r'C:\Users\hanmi\Documents\xiguabook\Transfusion.txt'

data = np.loadtxt(data_path, delimiter=',').astype(int)

X = data[:, :4]
y = data[:, 4]

m, n = X.shape

# normalization
X = (X - X.mean(0)) / X.std(0)

# shuffle
index = np.arange(m)
np.random.shuffle(index)

X = X[index]
y = y[index]

# 使用sklarn 中自带的api先
# k-10 cross validation
lr = linear_model.LogisticRegression(C=2)

score = cross_val_score(lr, X, y, cv=10)

print(score.mean())

# LOO
loo = LeaveOneOut()

accuracy = 0
for train, test in loo.split(X, y):
    lr_ = linear_model.LogisticRegression(C=2)
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]
    lr_.fit(X_train, y_train)

    accuracy += lr_.score(X_test, y_test)

print(accuracy / m)

# 两者结果几乎一样

# 自己写一个试试
# k-10
# 这里就没考虑最后几个样本了。
num_split = int(m / 10)
score_my = []
for i in range(10):
    lr_ = linear_model.LogisticRegression(C=2)
    test_index = range(i * num_split, (i + 1) * num_split)                  #这一步用于得到测试集的元素位置
    X_test_ = X[test_index]                                                 #得到对应得测试得数据，，， 一个是输入x  一个是结果值y
    y_test_ = y[test_index]

    X_train_ = np.delete(X, test_index, axis=0)                             # _train数据用于训练模型
    y_train_ = np.delete(y, test_index, axis=0)

    lr_.fit(X_train_, y_train_)                                             #模型的适应

    score_my.append(lr_.score(X_test_, y_test_))                            #利用测试集对测试集进行预测

print(np.mean(score_my))                                                    #应该是用于计算误差的

# LOO
score_my_loo = []
for i in range(m):
    lr_ = linear_model.LogisticRegression(C=2)
    X_test_ = X[i, :]
    y_test_ = y[i]

    X_train_ = np.delete(X, i, axis=0)
    y_train_ = np.delete(y, i, axis=0)

    lr_.fit(X_train_, y_train_)

    score_my_loo.append(int(lr_.predict(X_test_.reshape(1, -1)) == y_test_))

print(np.mean(score_my_loo))

# 结果都是类似
