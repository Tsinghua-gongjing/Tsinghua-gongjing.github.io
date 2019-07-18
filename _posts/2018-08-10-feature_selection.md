---
layout: post
category: "machinelearning"
title:  "特征选择"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

### 1. sklearn特征选择

* 模块：sklearn.feature_selection
* 可进行特征选择和数据降维
* 提高模型的准确度或者增强在高维数据上的性能

### 2. 方法：移除低方差特征

* 函数：VarianceThreshold
* 原理：移除方差不满足一些阈值的特征。默认: 移除所有方差为0（取值不变）的特征。

例子：布尔值类型的的数据聚集，移除0或者1超过80%的特征：
注意：布尔值是伯努利随机变量，方差：$$Var[X] = p(1-p)$$，如果选择阈值为0.8，则方差阈值=0.8(1-0.8)=0.8*0.2=0.16。可以看到，第一列特征被移除：

```python
>>> from sklearn.feature_selection import VarianceThreshold
>>> X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
>>> sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
>>> sel.fit_transform(X)
array([[0, 1],
       [1, 0],
       [0, 0],
       [1, 1],
       [1, 0],
       [1, 1]])
```

### 3. 方法：单变量特征选择

原理：对单变量进行统计测试，以选取最好的特征，可作为评估器的预处理步骤

* SelectKBest：只保留得分最高的k个特征
* SelectPercentile：移除指定的最高得分百分比之外的特征
* 常见统计测试量：SelectFpr(假阳性率), SelectFdr(假发现率), SelectFwe(族系误差)
* GenericUnivariateSelect：评估器超参数选择以选择最好的单变量
* 回归：f_regression，mutual_info_regression
* 分类：chi2，f_classif，mutual_info_classif

```python
>>> from sklearn.datasets import load_iris
>>> from sklearn.feature_selection import SelectKBest
>>> from sklearn.feature_selection import chi2
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> X.shape
(150, 4)
>>> X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
>>> X_new.shape
(150, 2)
```

### 4. 方法：递归式特征消除

### 5. 方法：SelectFromModel+L1的线性回归选取

* L1正则化：模型变得系数，很多系数为0
* 降低数据维度
* 选择非0系数特征

```python
>>> from sklearn.svm import LinearSVC
>>> from sklearn.datasets import load_iris
>>> from sklearn.feature_selection import SelectFromModel
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> X.shape
(150, 4)
>>> lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
>>> model = SelectFromModel(lsvc, prefit=True)
>>> X_new = model.transform(X)
>>> X_new.shape
(150, 3)
``` 

### 6. 方法：SelectFromModel+树的选取

* 计算特征的相关性
* 消除不相关特征

```python
>>> from sklearn.ensemble import ExtraTreesClassifier
>>> from sklearn.datasets import load_iris
>>> from sklearn.feature_selection import SelectFromModel
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> X.shape
(150, 4)
>>> clf = ExtraTreesClassifier()
>>> clf = clf.fit(X, y)
>>> clf.feature_importances_  
array([ 0.04...,  0.05...,  0.4...,  0.4...])
>>> model = SelectFromModel(clf, prefit=True)
>>> X_new = model.transform(X)
>>> X_new.shape               
(150, 2)
```

### 7. 特征选取放在管道中

作为预处理的步骤：

```python
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)
```

### 参考

* 机器学习周志华：特征选择与稀疏学习
* [sklearn 中文](https://sklearn.apachecn.org/#/docs/14?id=_113-%e7%89%b9%e5%be%81%e9%80%89%e6%8b%a9)