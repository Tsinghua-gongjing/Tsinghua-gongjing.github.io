---
layout: post
category: "machinelearning"
title:  "[CS229] 08: Neural Networks - Representation"
tags: [python, machine learning]
---

## 08: Neural Networks - Representation

1. 非线性问题：线性不可分，增加各种特征使得可分。比如根据图片检测汽车（计算机视觉）。当特征空间很大时，逻辑回归不再适用，而神经网络则是一个更好的非线性模型。
2. 神经网络：想要模拟大脑（不同的皮层区具有不同的功能，如味觉、听觉、触觉等），上世纪80-90年代很流行，90年达后期开始没落，现在又很流行，解决很多实际的问题。
3. 神经网络：
   - cell body, input wires (dendrities, 树突), output wire (axon，轴突)
3. 逻辑单元：最简单的神经元。一个输入层，一个激活函数，一个输出层。
4. 神经网络：激活函数，权重矩阵：
   - 输入层，输出层，隐藏层
   - ai(j) - activation of unit i in layer j ![](http://www.holehouse.org/mlclass/08_Neural_Networks_Representation_files/Image%20[7].png)
5. 前向传播：向量化实现，使用向量表示每一层次的输出。
6. 使用神经网络实现逻辑符号（逻辑与、逻辑或，逻辑和）：
   - 实现的是逻辑，而非线性问题，所以神经网络能很好的用于非线性问题上。
   - 下面的是实现 XNOR （NOT XOR）：![](http://www.holehouse.org/mlclass/08_Neural_Networks_Representation_files/Image%20[17].png)
7. 多分类问题：one-vs-all