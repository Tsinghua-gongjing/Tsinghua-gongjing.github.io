---
layout: post
category: "machinelearning"
title:  "sklearn: 数据预处理2"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

- TOC
{:toc}

## 数据预处理

>关于数据的预处理，sklearn提供了一个专门的模块`sklearn.preprocessing`，可用于常规的预处理操作，详情可参见这里（[英文](https://scikit-learn.org/stable/modules/preprocessing.html)，[中文](https://sklearn.apachecn.org/#/docs/40?id=_53-%e9%a2%84%e5%a4%84%e7%90%86%e6%95%b0%e6%8d%ae)）。

### 标准化（standardization）或去均值（mean removal）和方差按比例缩放（variance scaling）

a. 为什么？
   - 常见要求
   - 如果特征不是（不像）标准正太分布（0均值和单位方差），表现可能较差
   - 目标函数：假设所有特征都是0均值并且具有同一阶数上的方差。如果某个特征的方差比其他特征大几个数量级，则会占据主导地位
   
b. 函数`scale`：数组的标准化

```python
from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X_train)

X_scaled
# array([[ 0.  ..., -1.22...,  1.33...],
#       [ 1.22...,  0.  ..., -0.26...],
#       [-1.22...,  1.22..., -1.06...]])

# 缩放后的数据具有零均值以及标准方差
X_scaled.mean(axis=0)
# array([0., 0., 0.])

X_scaled.std(axis=0)
# array([1., 1., 1.])
```

c. 类`StandardScaler`：
   - 在训练集上计算平均值和标准偏差 
   - 可在测试集上直接使用相同变换
   - 适用于pipeline的早起构建

```python
scaler = preprocessing.StandardScaler().fit(X_train)
scaler
# StandardScaler(copy=True, with_mean=True, with_std=True)

scaler.mean_                                      
# array([1. ..., 0. ..., 0.33...])

scaler.scale_                                       
# array([0.81..., 0.81..., 1.24...])

scaler.transform(X_train) 
# array([[ 0.  ..., -1.22...,  1.33...],
#       [ 1.22...,  0.  ..., -0.26...],
#       [-1.22...,  1.22..., -1.06...]])
```

   - 在训练集计算的缩放直接应用于测试集：

```python
X_test = [[-1., 1., 0.]]
scaler.transform(X_test)
# array([[-2.44...,  1.22..., -0.26...]])
```

d. 将特征缩放至特定范围
   - 给定的最大、最小值之间，通常[0,1]
   - 类`MinMaxScaler`

```python
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_train_minmax
# array([[0.5       , 0.        , 1.        ],
#       [1.        , 0.5       , 0.33333333],
#       [0.        , 1.        , 0.        ]])

# 应用于测试集
X_test = np.array([[ -3., -1.,  4.]])
X_test_minmax = min_max_scaler.transform(X_test)
X_test_minmax
# array([[-1.5       ,  0.        ,  1.66666667]])
```
   
   - 每个特征的最大绝对值转换为单位大小
   - 除以每个特征的最大值，转换为[-1,1]之间
   - 类`MaxAbsScaler`

```python
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_train_maxabs                # doctest +NORMALIZE_WHITESPACE^
# array([[ 0.5, -1. ,  1. ],
#       [ 1. ,  0. ,  0. ],
#       [ 0. ,  1. , -0.5]])


X_test = np.array([[ -3., -1.,  4.]])
X_test_maxabs = max_abs_scaler.transform(X_test)
X_test_maxabs                 
# array([[-1.5, -1. ,  2. ]])

max_abs_scaler.scale_  
# array([2.,  1.,  2.])
```

e. 缩放系数矩阵
   - 中心化会破坏原始的稀疏结构
   - `scale`,`StandardScaler`可以接受稀疏输入，构造时指定`with_mean=False`

f. 缩放含有离群值的数据
   - 此种情况下均值和方差缩放不是很好的选择
   - `robust_scale`, `RobustScaler`可作为替代

### 非线性变换

a. 常见可用：
   - 分位数转换：可是异常分布更平滑，受异常值影响小，但是使得特征间或内的关联失真，均匀分布在[0,1]
   - 幂函数转换：参数变换，目的是转换为接近高斯分布
   - 两者均基于特征的单调变换，保持了特征值的秩
   
b. 映射到均匀分布：
   - 分位数类：`QuantileTransformer`
   - 分位数函数：`quantile_transform`
   
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]) 
# array([ 4.3,  5.1,  5.8,  6.5,  7.9])

# 转换后接近于百分位数定义
np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100])
# array([ 0.00... ,  0.24...,  0.49...,  0.73...,  0.99... ])
```

c. 映射到高斯分布
   - 幂变换是一类参数化的单调变换
   - 将数据从任何分布映射到尽可能接近高斯分布，以便稳定方差和最小化偏斜。
   - `PowerTransformer`提供两种：`Yeo-Johnson transform`, `the Box-Cox transform`

```python
pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
X_lognormal = np.random.RandomState(616).lognormal(size=(3, 3))
X_lognormal                                         
# array([[1.28..., 1.18..., 0.84...],
#       [0.94..., 1.60..., 0.38...],
#       [1.35..., 0.21..., 1.09...]])

pt.fit_transform(X_lognormal)
# array([[ 0.49...,  0.17..., -0.15...],
#       [-0.05...,  0.58..., -0.57...],
#       [ 0.69..., -0.84...,  0.10...]])
```

	- 不同的原始分布经过变化，有的可以变为高斯分布，有的可能效果不太好 ![](https://sklearn.apachecn.org/docs/img/sphx_glr_plot_map_data_to_normal_0011.png)

### 归一化（normalization）

	- 缩放单个样本以具有单位范数的过程
	- 函数`normalize`：用于数组，`l1`或`l2`范式

```python
X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2')

X_normalized
# array([[ 0.40..., -0.40...,  0.81...],
#       [ 1.  ...,  0.  ...,  0.  ...],
#       [ 0.  ...,  0.70..., -0.70...]])
```

### 类别特征编码

a. 标称型特征：非连续的数值，可编码为整数
b. 类`OrdinalEncoder`：
	- 将类别特征值编码为一个新的整数型特征（0到num_category-1之间的一个数）
	- 但是这个数值不能直接使用，因为会被认为是有序的（实际是无序的）
	- 一般使用独热编码

```python
enc = preprocessing.OrdinalEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)  
# OrdinalEncoder(categories='auto', dtype=<... 'numpy.float64'>)

enc.transform([['female', 'from US', 'uses Safari']])
# array([[0., 1., 1.]])
```

c. 独热编码（dummy encoding）
   - 该类把每一个具有n_categories个可能取值的categorical特征变换为长度为n_categories的二进制特征向量，里面只有一个地方是1，其余位置都是0。

```python
enc = preprocessing.OneHotEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)  
# OneHotEncoder(categorical_features=None, categories=None, drop=None,
#       dtype=<... 'numpy.float64'>, handle_unknown='error',
#       n_values=None, sparse=True)

enc.transform([['female', 'from US', 'uses Safari'],
               ['male', 'from Europe', 'uses Safari']]).toarray()
# array([[1., 0., 0., 1., 0., 1.],
#        [0., 1., 1., 0., 0., 1.]]) 

# 在category_属性中找到，编码后是几维的
enc.categories_
# [array(['female', 'male'], dtype=object), array(['from Europe', 'from US'], dtype=object), array(['uses Firefox', 'uses Safari'], dtype=object)]             
```

可以显示的指定，某个特征需要被编码为几维的，最开始提供一个可能的取值集合，基于这个集合进行编码：

```python
genders = ['female', 'male']
locations = ['from Africa', 'from Asia', 'from Europe', 'from US']
browsers = ['uses Chrome', 'uses Firefox', 'uses IE', 'uses Safari']
enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers])
# Note that for there are missing categorical values for the 2nd and 3rd
# feature
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X) 


enc.transform([['female', 'from Asia', 'uses Chrome']]).toarray()
# array([[1., 0., 0., 1., 0., 0., 1., 0., 0., 0.]])
```

### 离散化

a. 离散化（discretization）：
   - 又叫量化(quantization) 或 装箱(binning)
   - 将连续特征划分为离散特征值
   
b. K-bins离散化：
   - 使用k个等宽的bins把特征离散化
   - 对每一个特征， bin的边界以及总数目在 fit过程中被计算出来
  
```python
X = np.array([[ -3., 5., 15 ],
              [  0., 6., 14 ],
              [  6., 3., 11 ]])
est = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal').fit(X)

est.transform(X)  
# array([[ 0., 1., 1.],
#       [ 1., 1., 1.],
#       [ 2., 0., 0.]])
```

d. 特征二值化
	- 将数值特征用阈值过滤得到布尔值的过程
	
```python
X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]

binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
binarizer


binarizer.transform(X)
# array([[1., 0., 1.],
#       [1., 0., 0.],
#       [0., 1., 0.]])

# 可使用不同的阈值
binarizer = preprocessing.Binarizer(threshold=1.1)
binarizer.transform(X)
# array([[0., 0., 1.],
#       [1., 0., 0.],
#       [0., 0., 0.]])
```

### 生成多项式特征

a. 为什么？
	- 添加非线性特征
	- 增加模型的复杂度
	- 常用：添加多项式

b. 生成多项式类`PolynomialFeatures`：
 
$$(X_1,X_2) => (1, X_1,X_2,X_1^2,X_1X_2,X_2^2)$$

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
X                                                 
# array([[0, 1],
#       [2, 3],
#       [4, 5]])

poly = PolynomialFeatures(2)
poly.fit_transform(X) 
# array([[ 1.,  0.,  1.,  0.,  0.,  1.],
#        [ 1.,  2.,  3.,  4.,  6.,  9.],
#       [ 1.,  4.,  5., 16., 20., 25.]]) 
```

$$(X_1,X_2,X_3) => (1, X_1,X_2,X_3,X_1X_2,X_1X_3,X_2X_3,X_1X_2X_3)$$

```python
X = np.arange(9).reshape(3, 3)
X                                                 
# array([[0, 1, 2],
#       [3, 4, 5],
#       [6, 7, 8]])

# 指定度，且只要具有交叉的项，像上面的自己平方的项不要了
poly = PolynomialFeatures(degree=3, interaction_only=True)
poly.fit_transform(X) 
# array([[  1.,   0.,   1.,   2.,   0.,   0.,   2.,   0.],
#       [  1.,   3.,   4.,   5.,  12.,  15.,  20.,  60.],
#       [  1.,   6.,   7.,   8.,  42.,  48.,  56., 336.]])                            
```

### 自定义转换器

类`FunctionTransformer`:

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p, validate=True)
X = np.array([[0, 1], [2, 3]])
transformer.transform(X)
# array([[0.        , 0.69314718],
#       [1.09861229, 1.38629436]])
```

