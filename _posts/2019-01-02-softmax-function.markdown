---
layout: post
category: "machinelearning"
title:  "Softmax function"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

### 概念

softmax函数（归一化的指数函数）："squashes"(maps) a K-dimensional vector z of arbitrary real values to a K-dimensional vector σ(z) of real values in the range (0, 1) that add up to 1 （from [wiki](https://en.wikipedia.org/wiki/Softmax_function)）。
 
   - 向量原来的每个值都转换为指数的概率值：$$\sigma(z)_{j}=\frac{e^{z_{j}}}{\sum^{K}_{k=1}e^{z_{k}}}$$。
   - 转换后的值是个概率值，在[0,1]之间；
   - 转换后的向量加和为1。
   - 下面是用代码举例子说明是怎么计算的：
   - ```python
>>> import math
>>> z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
>>> z_exp = [math.exp(i) for i in z]
>>> sum_z_exp = sum(z_exp)
>>> softmax = [i / sum_z_exp for i in z_exp]
>>> print([round(i, 3) for i in softmax])
[0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
```

---

### 应用

   - 神经网络的多分类问题，作为最后一层（输出层），转换为类别概率，如[下图](https://stats.stackexchange.com/questions/265905/derivative-of-softmax-with-respect-to-weights)所示，但是这个图里面`e`的下标`k`应该写错了位置，`k`应该是`z`的下标（一般最后基于概率值有一个独热编码，对应具体的类别）：![](https://i.stack.imgur.com/0rewJ.png)
   - 将某个值转换为激活概率，比如增强学习领域，此时的其公式为：$$P_{t}(a)=\frac{e^{\frac{q_{t}(a)}{T}}}{\sum^{n}_{i=1}e^{\frac{q_{t}(i)}{T}}}$$

---

### softmax vs logistic

   - 参考[这里：logistic函数和softmax函数](https://www.cnblogs.com/maybe2030/p/5678387.html)
   - logistic： 二分类问题，基于**多项式分布**
   - softmax：多分类问题，基于**伯努利分布**
   - 因此logistic是softmax函数的一个特例，就是当K=2时的情况。所以在逻辑回归那里，也有softmax regression（多元逻辑回归）用于多分类问题，我在[这里](https://tsinghua-gongjing.github.io/posts/CS229-06-logistic-regression.html)也记录了一点。
   - 在多分类里面，也可以使用多个one-vs-all的逻辑回归，达到多元回归的目的，这种操作和直接的softmax回归有什么不同？softmax回归输出的类是唯一互斥的，但是多个逻辑回归的输出类别不一定是互斥的。

---







