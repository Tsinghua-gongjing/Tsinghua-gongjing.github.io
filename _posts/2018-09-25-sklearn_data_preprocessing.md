---
layout: post
category: "machinelearning"
title:  "sklearn: 数据预处理"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

这个来源于[这里](https://github.com/MLEveryday/100-Days-Of-ML-Code/blob/master/Code/Day%201_Data_Preprocessing.md)，英文版本在[这里](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day%201_Data%20PreProcessing.md)

如下图所示，通过6步完成数据预处理操作：

[![Day1_sklearn_data_preprocessing.jpg](https://i.loli.net/2019/07/12/5d28227f77d1689548.jpg)](https://i.loli.net/2019/07/12/5d28227f77d1689548.jpg)

此例用到的[数据](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/datasets/Data.csv)，[代码](https://github.com/MLEveryday/100-Days-Of-ML-Code/blob/master/Code/Day%201_Data_Preprocessing.py)。

### 第1步：导入库

```Python
import numpy as np
import pandas as pd
```

### 第2步：导入数据集

数据集通常是.csv格式。CSV文件以文本形式保存表格数据。文件的每一行是一条数据记录。我们使用Pandas的read_csv方法读取本地csv文件为一个数据帧。然后，从数据帧中制作自变量和因变量的矩阵和向量。

查看数据集，前3列是特征，最后1列是否购买（标签）：

```bash
gongjing@hekekedeiMac ..yn/github/100-Days-Of-ML-Code/datasets (git)-[master] % cat Data.csv
Country,Age,Salary,Purchased
France,44,72000,No
Spain,27,48000,Yes
Germany,30,54000,No
Spain,38,61000,No
Germany,40,,Yes
France,35,58000,Yes
Spain,,52000,No
France,48,79000,Yes
Germany,50,83000,No
France,37,67000,Yes%
```

读取数据集：

```python
dataset = pd.read_csv('Data.csv')//读取csv文件
X = dataset.iloc[ : , :-1].values//.iloc[行，列]
Y = dataset.iloc[ : , 3].values  // : 全部行 or 列；[a]第a行 or 列
                                 // [a,b,c]第 a,b,c 行 or 列
```

查看读取的数据集：

```python
print("X")
print(X)
print("Y")
print(Y)

X
[['France' 44.0 72000.0]
 ['Spain' 27.0 48000.0]
 ['Germany' 30.0 54000.0]
 ['Spain' 38.0 61000.0]
 ['Germany' 40.0 nan]
 ['France' 35.0 58000.0]
 ['Spain' nan 52000.0]
 ['France' 48.0 79000.0]
 ['Germany' 50.0 83000.0]
 ['France' 37.0 67000.0]]
Y
['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']
```

### 第3步：处理丢失数据

我们得到的数据很少是完整的。数据可能因为各种原因丢失，为了不降低机器学习模型的性能，需要处理数据。我们可以用整列的平均值或中间值替换丢失的数据。我们用sklearn.preprocessing库中的Imputer类完成这项任务。

```python
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
# 对第2、3列进行imputing，因为这两个特征是数值项
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
```

### 第4步：解析分类数据

分类数据指的是含有标签值而不是数字值的变量。取值范围通常是固定的。例如"Yes"和"No"不能用于模型的数学计算，所以需要解析成数字。为实现这一功能，我们从sklearn.preprocessing库导入LabelEncoder类。

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
```

### 创建虚拟变量

```python
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
```

查看编码之后的数据。从这里可以看到，原来的第一列是国家（类别特征），包含3个：France、Spain、Germany，在做了转换之后，特征多了两个，相当于原来的1个特征现在用3个特征来表示：

```python
print("Step 4: Encoding categorical data")
print("X")
print(X)
print("Y")
print(Y)

X
[[1.00000000e+00 0.00000000e+00 0.00000000e+00 4.40000000e+01
  7.20000000e+04]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00 2.70000000e+01
  4.80000000e+04]
 [0.00000000e+00 1.00000000e+00 0.00000000e+00 3.00000000e+01
  5.40000000e+04]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00 3.80000000e+01
  6.10000000e+04]
 [0.00000000e+00 1.00000000e+00 0.00000000e+00 4.00000000e+01
  6.37777778e+04]
 [1.00000000e+00 0.00000000e+00 0.00000000e+00 3.50000000e+01
  5.80000000e+04]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00 3.87777778e+01
  5.20000000e+04]
 [1.00000000e+00 0.00000000e+00 0.00000000e+00 4.80000000e+01
  7.90000000e+04]
 [0.00000000e+00 1.00000000e+00 0.00000000e+00 5.00000000e+01
  8.30000000e+04]
 [1.00000000e+00 0.00000000e+00 0.00000000e+00 3.70000000e+01
  6.70000000e+04]]
Y
[0 1 0 0 1 1 0 1 0 1]
```

### 第5步：拆分数据集为训练集合和测试集合

把数据集拆分成两个：一个是用来训练模型的训练集合，另一个是用来验证模型的测试集合。两者比例一般是80:20。我们导入sklearn.model_selection库中的train_test_split()方法。

```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
```

查看查分后的数据：

```python
print("Step 5: Splitting the datasets into training sets and Test sets")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)

Step 5: Splitting the datasets into training sets and Test sets
X_train
[[  0.00000000e+00   1.00000000e+00   0.00000000e+00   4.00000000e+01
    6.37777778e+04]
 [  1.00000000e+00   0.00000000e+00   0.00000000e+00   3.70000000e+01
    6.70000000e+04]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00   2.70000000e+01
    4.80000000e+04]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00   3.87777778e+01
    5.20000000e+04]
 [  1.00000000e+00   0.00000000e+00   0.00000000e+00   4.80000000e+01
    7.90000000e+04]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00   3.80000000e+01
    6.10000000e+04]
 [  1.00000000e+00   0.00000000e+00   0.00000000e+00   4.40000000e+01
    7.20000000e+04]
 [  1.00000000e+00   0.00000000e+00   0.00000000e+00   3.50000000e+01
    5.80000000e+04]]
X_test
[[  0.00000000e+00   1.00000000e+00   0.00000000e+00   3.00000000e+01
    5.40000000e+04]
 [  0.00000000e+00   1.00000000e+00   0.00000000e+00   5.00000000e+01
    8.30000000e+04]]
Y_train
[1 1 1 0 1 0 0 1]
Y_test
[0 0]
```

### 第6步：特征量化

大部分模型算法使用两点间的欧氏距离表示，但此特征在幅度、单位和范围姿态问题上变化很大。在距离计算中，高幅度的特征比低幅度特征权重更大。可用特征标准化或Z值归一化解决。导入sklearn.preprocessing库的StandardScalar类

```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
```

```python
print("Step 6: Feature Scaling")
print("X_train")
print(X_train)
print("X_test")
print(X_test)

Step 6: Feature Scaling
X_train
[[-1.          2.64575131 -0.77459667  0.26306757  0.12381479]
 [ 1.         -0.37796447 -0.77459667 -0.25350148  0.46175632]
 [-1.         -0.37796447  1.29099445 -1.97539832 -1.53093341]
 [-1.         -0.37796447  1.29099445  0.05261351 -1.11141978]
 [ 1.         -0.37796447 -0.77459667  1.64058505  1.7202972 ]
 [-1.         -0.37796447  1.29099445 -0.0813118  -0.16751412]
 [ 1.         -0.37796447 -0.77459667  0.95182631  0.98614835]
 [ 1.         -0.37796447 -0.77459667 -0.59788085 -0.48214934]]
X_test
[[ 0.  0.  0. -1. -1.]
 [ 0.  0.  0.  1.  1.]]
```
