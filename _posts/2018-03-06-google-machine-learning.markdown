---
layout: post
category: "machinelearning"
title:  "Google machine learning"
tags: [python, machine learning, google]
---

- TOC
{:toc}

最近，[google](https://developers.google.com/machine-learning/crash-course)在其开发网站上，公开了用于内部人员进行机器学习培训的材料，可以快速帮助了解机器学习及其框架TensorFlow。量子位提供了一个相关材料的连接[（别翻墙了，谷歌机器学习速成课25讲视频全集在此）](https://mp.weixin.qq.com/s?__biz=MzIzNjc1NzUzMw==&mid=2247495096&idx=1&sn=cb25eec7088e96f416bc1df2a6c2df10&chksm=e8d05acadfa7d3dca298cef87ed9bf836a81d9501be6210cac5c9a2f6fdc1a4dc81b10348229&mpshare=1&scene=23&srcid=0304E57F5bWLPOV7AUtjemDr#rd)。近期会学习这个系列的材料，做一点后续的笔记。


## 课程概览

### 机器学习概念

####  简介

#### 框架处理


1. 介绍了基本框架（监督式机器学习）和及一些术语
2. 基本术语：
   - 标签：要预测的真实事物（y）
   - 特征：描述数据的输入变量（xi）。特征尽量是可量化的，比如鞋码、点击次数等，美观程度等不可量化，不适合作为特征。
   - 样本：数据的特定实例
   - 有标签样本：训练模型；无标签样本：对新数据做预测
3. 模型：将样本映射到预测标签
   - 回归模型：预测连续值
   - 分类模型：预测离散值

#### 深入了解机器学习

1. 介绍了线性回归，房屋面积预测销售价格的例子，引出如何评价线性回归的好坏。
2. 模型训练：通过有标签样本来学习（确定）所有权重和偏差的理想值
3. 监督式学习（经验风险最小化）：检查多个样本并尝试找出可最大限度地减少损失的模型
4. 损失：
   - L2loss（平方误差/平方损失）：预测值和标签值之差的平方
   - 均方误差 (MSE) ：每个样本的平均平方损失，平方损失之和/样本数量

#### 降低损失

1. 训练模型需要reducing loss，迭代方法很常用。
2. 迭代：根据输入和模型的随机初始值计算损失值，然后更新模型参数值，不断循环，使得损失值达到最小（试错过程）。当损失不再变化或者变化极其缓慢，可以说模型区域收敛。
3. 梯度下降：上述迭代过程存在更新模型参数的步骤，使用此方法快速寻找达到最小损失的参数（不可能把所有的参数都尝试一遍）。
4. 梯度：损失曲线在对应参数处的梯度（偏导数：有大小和方向），负梯度是对应梯度下降的。
5. 如何选取下一个点？
  - 下降梯度值 x 学习速率（步长）
6. 如果知道损失函数梯度较小，可以使用较大的学习速率，以较快的达到最小损失。
7. 优化学习速率，理解通过调节不同的速率，使得学习效率能够收敛达到最高。在降低损失的方向选取小步长（因为想要精准的达到最低损失）。小步长：梯度步长。这种优化策略：梯度下降法。
8. 权重初始化：凸形问题，可任意点开始，因为只有一个最低点；非凸形：有多个最低点，很大程度决定于起始值。
9. 批量：单次迭代中计算梯度的样本总数。一般是总的样本，但是海量数据过于庞大。
10. 随机梯度下降（SGD）：一次抽取一个样本
11. 小批量梯度下降：每批10-1000个样本，使得估算均值接近整体，且计算时间可接受

#### 使用TF的基本步骤

1. TF(TensorFlow) API:
  - 面向对象的高级API：estimator
  - 库：tf.layers, tf.losses, tf.metrics
  - 可封装C++内核的指令：TensorFlow python/C++
  - 多平台：CPU/GPU/TPU

```python
# tf.estimator API

import tensorflow as tf

# set up a classifier
classifier = tf.estimator.LinearClassifier()

# Train the model on some example data.
# what does steps mean here?
classifier.train(input_fn=train_input_fn, steps=2000)

# Use it to predict.
predictions = classifier.predict(input_fn=predict_input_fn)
```

#### 泛化
#### 训练集和测试集
#### 验证
#### 表示法
#### 特征组合
11. 正则化：简单行
12. 逻辑回归
13. 分类
14. 正则化：稀疏性
15. 神经网络简介
16. 训练神经网络
17. 多类别神经网络
18. 嵌入

### 机器学习工程

1. 生产环境机器学习系统
2. 静态训练与动态训练
3. 静态推理与动态推理
4. 数据依赖关系

### 机器学习现实世界应用示例

1. 癌症预测
2. 18世纪文学
3. 现实世界应用准则

### 总结

1. 后续步骤

