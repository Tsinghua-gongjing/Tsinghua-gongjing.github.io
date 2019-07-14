---
layout: post
category: "machinelearning"
title:  "模型评估与选择2"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

### 交叉验证：评估模型的表现

#### 1. 计算交叉验证的指标（分数等）：函数`cross_val_score`

```python
>>> from sklearn.model_selection import cross_val_score
>>> clf = svm.SVC(kernel='linear', C=1)

# cv=5指重复5次交叉验证，默认每次是5fold
>>> scores = cross_val_score(clf, iris.data, iris.target, cv=5)
>>> scores                                              
array([0.96..., 1.  ..., 0.96..., 0.96..., 1.        ])

# 计算平均得分、95%置信区间
>>> print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Accuracy: 0.98 (+/- 0.03)
```

#### 2. 可通过`scoring`参数指定计算特定的分数值

```python
>>> from sklearn import metrics
>>> scores = cross_val_score(
...     clf, iris.data, iris.target, cv=5, scoring='f1_macro')
>>> scores                                              
array([0.96..., 1.  ..., 0.96..., 0.96..., 1.        ])
```

#### 3. 自行设定交叉验证器，更好的控制交叉验证过程

默认的情况下，`cross_val_score`不能直接指定划分的具体细节（比如训练集测试集的比例，初始化值，重复次数）等，可以自行设定好之后，传给其参数`cv`，这样能够获得更好的控制：

```python
>>> from sklearn.model_selection import ShuffleSplit
>>> n_samples = iris.data.shape[0]
>>> cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
>>> cross_val_score(clf, iris.data, iris.target, cv=cv)  
array([0.977..., 0.977..., 1.  ..., 0.955..., 1.        ])
```

#### 4. 获得多个度量值

默认的，`cross_val_score`只能计算一个类型的分数，要想获得多个度量值，可用函数`cross_validate`:

```python
>>> from sklearn.model_selection import cross_validate
>>> from sklearn.metrics import recall_score
>>> scoring = ['precision_macro', 'recall_macro']
>>> clf = svm.SVC(kernel='linear', C=1, random_state=0)
>>> scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
...                         cv=5)

# 默认是运行和打分时间+测试集的指标
>>> sorted(scores.keys())
['fit_time', 'score_time', 'test_precision_macro', 'test_recall_macro']
>>> scores['test_recall_macro']                       
array([0.96..., 1.  ..., 0.96..., 0.96..., 1.        ])

# 可以指定return_train_score参数，同时返回训练集的度量指标值
>>> scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
...                         cv=5, return_train_score=True)
>>> sorted(scores.keys())                 
['fit_time', 'score_time', 'test_prec_macro', 'test_rec_macro',
 'train_prec_macro', 'train_rec_macro']
>>> scores['train_rec_macro']                         
array([0.97..., 0.97..., 0.99..., 0.98..., 0.98...])
```

#### 5. 数据切分：k-fold & repeat k-fold

通常使用k-fold对数据进行切分：

```python
>>> import numpy as np
>>> from sklearn.model_selection import KFold

>>> X = ["a", "b", "c", "d"]
>>> kf = KFold(n_splits=2)
>>> for train, test in kf.split(X):
...     print("%s  %s" % (train, test))
[2 3] [0 1]
[0 1] [2 3]
```

一般k-fold需要重复多次，下面是重复2次2-fold：

```python
>>> import numpy as np
>>> from sklearn.model_selection import RepeatedKFold
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> random_state = 12883823
>>> rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
>>> for train, test in rkf.split(X):
...     print("%s  %s" % (train, test))
...
[2 3] [0 1]
[0 1] [2 3]
[0 2] [1 3]
[1 3] [0 2]
```

#### 6. 留一法、留P法 交叉验证

每次`一个`样本用于测试：

```python
>>> from sklearn.model_selection import LeaveOneOut

>>> X = [1, 2, 3, 4]
>>> loo = LeaveOneOut()
>>> for train, test in loo.split(X):
...     print("%s  %s" % (train, test))
[1 2 3] [0]
[0 2 3] [1]
[0 1 3] [2]
[0 1 2] [3]
```

每次`P个`样本用于测试：

```python
>>> from sklearn.model_selection import LeavePOut

>>> X = np.ones(4)
>>> lpo = LeavePOut(p=2)
>>> for train, test in lpo.split(X):
...     print("%s  %s" % (train, test))
[2 3] [0 1]
[1 3] [0 2]
[1 2] [0 3]
[0 3] [1 2]
[0 2] [1 3]
[0 1] [2 3]
```

作为一般规则，大多数作者和经验证据表明， 5- 或者 10- 交叉验证应该优于 留一法交叉验证。

#### 7. 随机排列交叉验证

函数`ShuffleSplit`，先将样本打散，再换分为一对训练测试集合，这也是上面的`3. 自行设定交叉验证器，更好的控制交叉验证过程`提到的，更好的控制交叉验证所用到的函数：

```python
>>> from sklearn.model_selection import ShuffleSplit
>>> X = np.arange(5)
>>> ss = ShuffleSplit(n_splits=3, test_size=0.25,
...     random_state=0)
>>> for train_index, test_index in ss.split(X):
...     print("%s  %s" % (train_index, test_index))
...
[1 3 4] [2 0]
[1 4 3] [0 2]
[4 0 2] [1 3]
```

#### 8. 具有标签的分层交叉验证

分层k-fold：每个fold里面，各类别样本比例大致相当：

```python
>>> from sklearn.model_selection import StratifiedKFold

>>> X = np.ones(10)
>>> y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
>>> skf = StratifiedKFold(n_splits=3)
>>> for train, test in skf.split(X, y):
...     print("%s  %s" % (train, test))
[2 3 6 7 8 9] [0 1 4 5]
[0 1 3 4 5 8 9] [2 6 7]
[0 1 2 4 5 6 7] [3 8 9]
```

#### 9. 分组数据的交叉验证

分组k-fold：同一组在测试和训练中不被同时表示，某一个组的数据，要么出现在训练集，要么出现在测试集：

```python
>>> from sklearn.model_selection import GroupKFold

>>> X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
>>> y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
>>> groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

>>> gkf = GroupKFold(n_splits=3)
>>> for train, test in gkf.split(X, y, groups=groups):
...     print("%s  %s" % (train, test))
[0 1 2 3 4 5] [6 7 8 9]
[0 1 2 6 7 8 9] [3 4 5]
[3 4 5 6 7 8 9] [0 1 2]
```

分组：留一组

```python
>>> from sklearn.model_selection import LeaveOneGroupOut

>>> X = [1, 5, 10, 50, 60, 70, 80]
>>> y = [0, 1, 1, 2, 2, 2, 2]
>>> groups = [1, 1, 2, 2, 3, 3, 3]
>>> logo = LeaveOneGroupOut()
>>> for train, test in logo.split(X, y, groups=groups):
...     print("%s  %s" % (train, test))
[2 3 4 5 6] [0 1]
[0 1 4 5 6] [2 3]
[0 1 2 3] [4 5 6]
```

分组：留P组：

```python
>>> from sklearn.model_selection import LeavePGroupsOut

>>> X = np.arange(6)
>>> y = [1, 1, 1, 2, 2, 2]
>>> groups = [1, 1, 2, 2, 3, 3]
>>> lpgo = LeavePGroupsOut(n_groups=2)
>>> for train, test in lpgo.split(X, y, groups=groups):
...     print("%s  %s" % (train, test))
[4 5] [0 1 2 3]
[2 3] [0 1 4 5]
[0 1] [2 3 4 5]
```

#### 10. 时间序列分割

```python
>>> from sklearn.model_selection import TimeSeriesSplit

>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> tscv = TimeSeriesSplit(n_splits=3)
>>> print(tscv)  
TimeSeriesSplit(max_train_size=None, n_splits=3)
>>> for train, test in tscv.split(X):
...     print("%s  %s" % (train, test))
[0 1 2] [3]
[0 1 2 3] [4]
[0 1 2 3 4] [5]
```

### 参考

* [模型选择和评估 @sklearn 中文版](https://sklearn.apachecn.org/#/docs/29?id=_3-%e6%a8%a1%e5%9e%8b%e9%80%89%e6%8b%a9%e5%92%8c%e8%af%84%e4%bc%b0)
* [Model selection and evaluation @sklearn](https://scikit-learn.org/stable/model_selection.html)