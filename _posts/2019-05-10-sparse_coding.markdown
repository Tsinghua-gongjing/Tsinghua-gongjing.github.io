---
layout: post
category: "machinelearning"
title:  "稀疏编码"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## 概述

稀疏编码（Sparse Coding）：又叫字典学习（Dictionary Learning）：属于一种无监督学习方法，寻求一组超完备的基，更高效的线性表示数据集。

<!-- more -->

在实际的应用中，稀疏表示的场景很多。比如对于图片，可以学习一组基向量，然后使用这些基向量的线性组合去表示图片。比如下图中，学习了64个基向量，然后用这些基向量去表示一个图片，在表示时其实只有3个是有值的，其他的基向量贡献都为0：

![](https://img-my.csdn.net/uploads/201304/09/1365483386_5095.jpg)

### 知识点

1. PCA：找一组特征向量，尽可能的表征原始数据，这里的基向量数目小于或等于原始数据的维度。但是在稀疏编码中，是寻找超完备向量，所以其基向量的数目远大于输入向量的维度。
2. 代价函数：$$\begin{align}
\text{minimize}_{a^{(j)}_i,\mathbf{\phi}_{i}} \sum_{j=1}^{m} \left|\left| \mathbf{x}^{(j)} - \sum_{i=1}^k a^{(j)}_i \mathbf{\phi}_{i}\right|\right|^{2} + \lambda \sum_{i=1}^{k}S(a^{(j)}_i)
\end{align}$$ 
   - m：输入向量的数目
   - X：输入向量
   - $$\phi$$：基向量
   - $$a_i\phi_i$$：向量x的稀疏表示
   - 【第1项】重构项（reconstruction term），迫使稀疏编码算法为输入向量提供一个高拟合度的线性表达式
   - 【第2项】稀疏惩罚项（sparsity penalty term），使输入向量的表达式变得“稀疏”，也就是系数向量a变得稀疏
   - $$\lambda$$：控制这两项式子的相对重要性
3. 训练：
   - 基于上面的目标函数进行迭代，求解最小损失时的$$a$$和$$\phi$$。因为有两个参数，所以类似于EM求解。
   - 固定$$\phi_k$$，然后调整$$a_k$$，使得上式，即目标函数最小（即解LASSO问题）。
   - 固定住$$a_k$$，调整$$\phi_k$$，使得上式，即目标函数最小（即解凸QP问题）。
   - 不断迭代，直至收敛。这样就可以得到一组可以良好表示这一系列x的基($$\phi$$)，也就是字典。
4. 表示（coding）：给定一个新的图片x，由上面得到的字典，通过解一个LASSO问题得到稀疏向量a。这个稀疏向量就是这个输入向量x的一个稀疏表达了。
   - 虽然根据样本训练学习得到了一组基向量，但是对于表示新样本的时候，让然需要优化该样本的系数$$a$$，所以其计算速度很慢。![](https://img-my.csdn.net/uploads/201304/09/1365483467_1398.jpg) ![](https://img-my.csdn.net/uploads/201304/09/1365483491_9524.jpg)
5. 为什么需要稀疏表示：
   - 特征的自动选择：无关的特征其权重都为0
   - 可解释性：只有少数不为0的权重说明此基向量是有作用的

## 参考

* [机器学习——字典学习/稀疏编码学习笔记@知乎](https://zhuanlan.zhihu.com/p/26015351)
* [稀疏编码Sparse coding](https://blog.csdn.net/LK274857347/article/details/76864828)
* [UFLDL笔记 - Sparse Coding（稀疏编码）](https://blog.csdn.net/walilk/article/details/78175912)
* [Sparse Coding @UFLDL](http://ufldl.stanford.edu/tutorial/unsupervised/SparseCoding/)





