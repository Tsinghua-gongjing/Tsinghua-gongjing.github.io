---
layout: post
category: "machinelearning"
title:  "sklearn: 教程"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## 目录

- TOC
{:toc}

### 1. 机器学习简介

---

#### 问题设置

* 考虑一系列数据，预测未知数据的属性
* 样本是多个属性的数据，则说有许多属性或者特征
* 监督学习：数据带有想要预测的结果值属性
	* 分类
	* 回归
* 无监督学习：数据是没有任何目标值的一组输入向量组成
	* 聚类：发现数据中彼此类似的示例
	* 密度估计：确定输入空间内的数据分布
	* 降维：将高维数据投射到二维或者三位进行可视化
* 训练集：从中学习数据的属性
* 测试集：测试学习到的性质

---

#### 加载示例数据集

具体参考[sklearn: 数据集加载](https://tsinghua-gongjing.github.io/posts/sklearn_dataset.html)

```python
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> digits = datasets.load_digits()

>>> print(digits.data)  
>>> digits.target

```

---

#### 学习和预测

* 基于训练数据，拟合一个估计器，预测未知的数据所属类别

```python
>>> from sklearn import svm

# 设置
>>> clf = svm.SVC(gamma=0.001, C=100.)

# 学习
>>> clf.fit(digits.data[:-1], digits.target[:-1]) 

# 预测
>>> clf.predict(digits.data[-1:])
```

---

#### 模型保存和加载

```python
>>> import pickle

# 保存
>>> s = pickle.dumps(clf)

# 重新加载
>>> clf2 = pickle.loads(s)

# 预测新数据
>>> clf2.predict(X[0:1])
array([0])
>>> y[0]
0
```

---

#### 再次训练和更新参数

* 同一个估计器，如果多次被调用，**超参数**存在更新，那么训练的到的模型也会被更新

```python
>>> import numpy as np
>>> from sklearn.datasets import load_iris
>>> from sklearn.svm import SVC
>>> X, y = load_iris(return_X_y=True)

# 第一次训练，得到一个模型
>>> clf = SVC()
>>> clf.set_params(kernel='linear').fit(X, y)  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
>>> clf.predict(X[:5])
array([0, 0, 0, 0, 0])

# 第二次训练，得到一个模型
# 直接指定模型需要更改的超参数
>>> clf.set_params(kernel='rbf', gamma='scale').fit(X, y)  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
>>> clf.predict(X[:5])
array([0, 0, 0, 0, 0])
```

---

#### 多分类和多标签拟合

如果预测的y是属于多类别的，比如有0，1，2三种类别，有两种方式进行预测：

* 一种是直接预测类别，输出为0，1，2其中的一个（也就是一种类别）
* 一种是先对y进行二值化（one-hot），比如这里就是一个1x3的数组

```python
>>> from sklearn.svm import SVC
>>> from sklearn.multiclass import OneVsRestClassifier
>>> from sklearn.preprocessing import LabelBinarizer

# 直接预测，每个样本是某一类
>>> X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
>>> y = [0, 0, 1, 1, 2]

# 先对y进行二值化编码，预测值是二值编码的形式
>>> classif = OneVsRestClassifier(estimator=SVC(random_state=0))
>>> classif.fit(X, y).predict(X)
array([0, 0, 1, 1, 2])

>>> y = LabelBinarizer().fit_transform(y)
>>> classif.fit(X, y).predict(X)
array([[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0],
       [0, 0, 0],  # 这里为0，表示预测的结果不匹配训练集合的任一标签
       [0, 0, 0]])
```

类似的，多标签的（一个样本属于多个类别，比如一个图片同时包含猫和狗）也可以先对y进行标签化，再训练预测：

```python
>> from sklearn.preprocessing import MultiLabelBinarizer
>> y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
>> y = MultiLabelBinarizer().fit_transform(y)
>> classif.fit(X, y).predict(X)
array([[1, 1, 0, 0, 0],
       [1, 0, 1, 0, 0],
       [0, 1, 0, 1, 0],
       [1, 0, 1, 1, 0],
       [0, 0, 1, 0, 1]])
# 这里有0-4共5个类别，所以是一个1x5的数组
# 如果某个样本属于多个类别，相应类别值为1
```

---

### 2. 统计学习：监督学习

---

#### k近邻分类器

---

#### 维度惩罚（curse of dimensionality）

* 高维情形下出现的数据样本稀疏，距离计算困难（算法的空间可能复杂度指数上升）等问题
* 解决方法一般是降维和特征选择，常见的降维方法有PCA，embedding等

---

#### 线性回归

* 最简单的拟合线性模型形式，是通过调整数据集的一系列参数令残差平方和尽可能小

---

#### 稀疏

---

#### 分类

---

#### 支持向量机

* 判别模型
* 找到样例的一个组合来构建一个两类之间最大化的平面
* 参数C：正则化

---

#### 使用核

* 特征空间内，数据不总是线性可分的
* 多项式或者非线性的进行拟合

---

### 3. 统计学习：模型选择

---

#### 交叉验证

---

#### 网格搜索

---

### 4. 统计学习：无监督学习

---

#### 聚类

* kmeans聚类：结果不能完全对应真实情况
* 合适的聚类数量难以选取
* 初始化值的敏感性，陷入局部最优

---

#### 分层聚类：慎用

* Agglomerative（聚合） - 自底向上的方法: 初始阶段，每一个样本将自己作为单独的一个簇，聚类的簇以最小化距离的标准进行迭代聚合。当感兴趣的簇只有少量的样本时，该方法是很合适的。如果需要聚类的 簇数量很大，该方法比K_means算法的计算效率也更高。
* Divisive（分裂） - 自顶向下的方法: 初始阶段，所有的样本是一个簇，当一个簇下移时，它被迭代的进 行分裂。当估计聚类簇数量较大的数据时，该算法不仅效率低(由于样本始于一个簇，需要被递归的进行 分裂)，而且从统计学的角度来讲也是不合适的。

---

#### 特征聚集

* 合并相似的维度（特征）

---

#### 分解：PCA、ICA

* 提取数据的主要成分

---

### 5. 利用管道把所有的放在一起

这个是使用面部特征进行人脸识别的例子：

```python
"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset:

================== ============ ======= ========== =======
                   precision    recall  f1-score   support
================== ============ ======= ========== =======
     Ariel Sharon       0.67      0.92      0.77        13
     Colin Powell       0.75      0.78      0.76        60
  Donald Rumsfeld       0.78      0.67      0.72        27
    George W Bush       0.86      0.86      0.86       146
Gerhard Schroeder       0.76      0.76      0.76        25
      Hugo Chavez       0.67      0.67      0.67        15
       Tony Blair       0.81      0.69      0.75        36

      avg / total       0.80      0.80      0.80       322
================== ============ ======= ========== =======

"""
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5, iid=False)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()

```

[![sklearn_put_it_all_face_recognize.png](https://i.loli.net/2019/07/21/5d34084f91f3791204.png)](https://i.loli.net/2019/07/21/5d34084f91f3791204.png)

---

### 6. 正确选择评估器

![](https://scikit-learn.org/stable/_static/ml_map.png)

---

### 参考

* [教程@sklearn 中文](https://sklearn.apachecn.org/#/docs/50)