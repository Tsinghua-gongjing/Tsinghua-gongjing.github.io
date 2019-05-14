---
layout: post
category: "machinelearning"
title:  "[CS229] 09: Neural Networks - Learning"
tags: [python, machine learning]
---

## 09: Neural Networks - Learning

1. 神经网络分类问题：
   - 二分类：输出为0或1
   - 多分类：比如有k个类别，则输出一个向量（长度为k，独热编码表示）
   - 损失函数：类比逻辑回归的损失函数
   - 逻辑回归的损失函数(对数损失+权重参数的正则项)：
   - ![](http://www.holehouse.org/mlclass/09_Neural_Networks_Learning_files/Image%20[2].png)
   - 神经网络的损失函数：
   - 注意1：输出有k个节点，所以需要对所有的节点进行计算
   - 注意2：第一部分，所有节点的平均逻辑对数损失
   - 注意3：第二部分，正则和（又称为weight decay），只不过是所有参数的
   - ![](http://www.holehouse.org/mlclass/09_Neural_Networks_Learning_files/Image%20[3].png)
2. 前向传播（forward propagation）：
   - 训练样本、结果已知
   - 每一层的权重可以用theta向量表示，这也是需要确定优化的参数
   - 每一层的激活函数已知
   - 就可以根据以上的数据和参数一层一层的计算每个节点的值，并与已知的值进行比较，构建损失函数
3. 反向传播（back propagation）：
   - 每一层的每个节点都会计算出一个值，但是这个值与真实值是有差异的，因此可以计算每个节点的错误。
   - 但是每个节点的真实值我们是不知道的，只知道最后的y值（输出值），因此需要从最后的输出值开始计算。
   - [这个文章: 一文弄懂神经网络中的反向传播法——BackPropagation](https://www.cnblogs.com/charlotte77/p/5629865.html)通过一个简单的3层网络的计算，演示了反向传播的过程，可以参考一下：
   - [![back_propagation1.jpeg](https://i.loli.net/2019/04/28/5cc48a65ea9a6.jpeg)](https://i.loli.net/2019/04/28/5cc48a65ea9a6.jpeg)
   - [![back_propagation2.jpeg](https://i.loli.net/2019/04/28/5cc48a65e77e1.jpeg)](https://i.loli.net/2019/04/28/5cc48a65e77e1.jpeg)
   
4. 神经网络学习：
   - [![neural_network_training.png](https://i.loli.net/2019/04/27/5cc401672148f.png)](https://i.loli.net/2019/04/27/5cc401672148f.png)