---
layout: post
category: "machinelearning"
title:  "Google ML excercises"
tags: [python, machine learning]
---

谷歌机器学习课程对应的[练习](https://developers.google.com/machine-learning/crash-course/exercises)。

## 预热

1. [Hello World](): 正常导入TF模块，其提供的notebook都是基于python3的，所以最好安装anaconda3，然后安装对应的tensorflow。
2. [TensorFlow编程概念]():
    - 张量(tensors)：任意维度的数组。可作为常量或变量存储在图中。
    - 标量：零维数组（零阶张量）；矢量：一维数组（一阶张量）；矩阵：二维数组（二阶张量）
    - 指令：创建、销毁和操控张量
    - 图（计算图、数据流图）：图数据结构，其节点是指令，边是张量。
    - 会话
    - 1）将常量、变量和指令整合到一个图中；2）在一个会话中评估这些常量、变量和指令。
3. [创建和操控张量]()：

- 矢量加法：类似于python里面的数组（numpy数组），比如元素级加法、元素翻倍等。 函数：`tf.add(x,y)`
- 广播：**元素级运算中的较小数组会增大到与较大数组具有相同的形状**。一般数学上只支持形状相同的张量进行元素级的运算，但是TF中借鉴Numpy的广播做法。

```python
twos = tf.constant([2, 2, 2, 2, 2, 2], dtype=tf.int32)
primes_doubled = primes * twos
print("primes_doubled:", primes_doubled)

# 输出值包含值和形状(shape)、类型信息
primes_doubled: tf.Tensor([ 4  6 10 14 22 26], shape=(6,), dtype=int32)
```

- 矩阵乘法：第一个举证的**列数**必须等于第二个矩阵的**行数**。函数：`tf.matmul`
- 张量变形：矩阵运算限制了矩阵的形状，因此需要频繁变化，可调用`tf.reshape`函数，改变形状或者阶数。
- 变量初始化和赋值：如果定义的是变量，需要进行初始化，这个值之后是可以更改(函数：`tf.assign`)的。

```python
v = tf.contrib.eager.Variable([3])
print(v)
tf.assign(v, [7])
print(v)

[3]
[7]
```

模拟投两个骰子10次，10x3的张量，第三列是前两列的和：
   - `random_uniform` ：随机选取
   - `tf.concat` ：合并张量

```python
die1 = tf.contrib.eager.Variable(
    tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))
die2 = tf.contrib.eager.Variable(
    tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))

dice_sum = tf.add(die1, die2)
resulting_matrix = (values=[die1, die2, dice_sum], axis=1)

print(resulting_matrix.numpy())

[[5 1 6]
 [4 5 9]
 [1 6 7]
 [2 3 5]
 [1 2 3]
 [5 4 9]
 [3 5 8]
 [3 2 5]
 [3 1 4]
 [6 3 9]]
```

4. [Pandas简介]()：
   - DataFrame：数据表格
   - Series：单一列
   - 利用索引随机化数据
   - 重建索引不包含时，会添加新行，并以`NaN`填充
   
```python
cities

City name	Population	Area square miles	Population density
0	San Francisco	852469	46.87	18187.945381
1	San Jose	1015785	176.53	5754.177760
2	Sacramento	485199	97.92	4955.055147

# 随机化index，然后重建索引
cities.reindex(np.random.permutation(cities.index))

# 索引超出范围不报错，新建行
cities.reindex([0, 4, 5, 2])
```
     

## 问题构建

[问题构建(Framing)](https://developers.google.com/machine-learning/crash-course/framing/check-your-understanding)理解：

1. 【监督式学习】：假设您想开发一种监督式机器学习模型来预测指定的电子邮件是“垃圾邮件”还是“非垃圾邮件”。以下哪些表述正确？
  - **有些标签可能不可靠。**
  - **未标记为“垃圾邮件”或“非垃圾邮件”的电子邮件是无标签样本。**
  - 我们将使用无标签样本来训练模型。
  - 主题标头中的字词适合做标签。
2. 【特征和标签】：假设一家在线鞋店希望创建一种监督式机器学习模型，以便为用户提供合乎个人需求的鞋子推荐。也就是说，该模型会向小马推荐某些鞋子，而向小美推荐另外一些鞋子。以下哪些表述正确？
  - 鞋的美观程度是一项实用特征。
  - 用户喜欢的鞋子是一种实用标签。
  - **“用户点击鞋子描述”是一项实用标签。**
  - **鞋码是一项实用特征。**

## 深入了解机器学习

[均方误差](https://developers.google.com/machine-learning/crash-course/descending-into-ml/check-your-understanding):

[![MSE.jpeg](https://i.loli.net/2019/04/12/5cb0275977e78.jpeg)](https://i.loli.net/2019/04/12/5cb0275977e78.jpeg)

## 降低损失

## 使用TensorFlow的起始步骤

## 训练集和测试集

## 验证

## 表示法

## 特征组合

## 简化正则化

## 分类

## 稀疏正则化

## 神经网络简介

## 训练神经网络

## 多类别神经网络

## 嵌套

## 静态训练和动态训练

## 静态推理和动态推理

## 数据依赖关系



