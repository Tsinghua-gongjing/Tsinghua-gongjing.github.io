---
layout: post
category: "machinelearning"
title:  "【2-2】深度学习的算法优化"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

### mini-batch梯度下降

* 优化算法：神经网络运行更快
* mini-batch vs batch：
	* batch：对**整个训练集**执行梯度下降
	* mini-batch：把训练集分割为小一点的子集，每次对其中**一个子集**执行梯度下降 ![](http://www.ai-start.com/dl2017/images/ef8c62ba3c82cb37e6ed4783e7717a8d.png)
	* 比如有5000000个样本，每1000个样本作为一个子集，那么可得到5000个子集
	* 用括号右上标表示：$$X^{\{1\}},...,X^{\{5000\}}$$
* 训练：
	* 训练和之前的batch梯度下降一致，只不过现在的样本是在每一个子集上进行 ![](http://www.ai-start.com/dl2017/images/0818dc0f8b7b8c1703d0c596f6027728.png)
	* 对于每一个batch，前向传播计算，计算Z，A，损失函数，再反向传播，计算梯度，更新参数 =》 这是完成一个min-batch样本的操作，比如这里是1000个样本
	
	* 遍历完所有的batch（这里是5000），就完成了一次所有样本的遍历，称为“一代”，也就是一个epoc
	* batch：一次遍历训练集做一个梯度下降
	* mini-batch：一次遍历训练集做batch-num个（这里是5000）梯度下降
	* 运行会更快

---

### 理解mini-batch梯度下降？

* 成本函数比较？
	* 两者的成本函数如下：![](http://www.ai-start.com/dl2017/images/b5c07d7dec7e54bed73cdcd43e79452d.png)
	* 可以看到，整体的趋势都是随着迭代次数增加在不断降低的
	* 但是batch情况下是单调下降的，而mini-batch则会出现波动性，有的mini-batch是上升的
	* 为什么有的mini-batch是上升的？有的是比较难计算的min-batch，会导致成本变高（那为啥还要min-batch？因为快呀！！！）。
* mini-batch size：
	* 看两个极端情况
	* size=m：batch梯度下降
	* size=1：随机梯度下降 ![](http://www.ai-start.com/dl2017/images/2181fbdc47d4d28840c3da00295548ea.png)
	
	* 收敛情况不同：
	* batch梯度下降：会收敛到最小，但是比较慢【蓝色】
	* 随机梯度下降：不会收敛到最小，在最小附近波动【紫色】
	* 合适的mini-batch梯度下降：较快的速度收敛到最小【绿色】![](http://www.ai-start.com/dl2017/images/bb2398f985f57c5d422df3c71437e5ea.png)

	* 小训练集：m<2000，使用batch梯度下降即可
	* 大训练集：mini-batch一般选为2的n次方的数目，比如64，128，256。64-512比较常见。

---

### 指数加权平均

* 还有一些比mini-batach梯度下降更快的算法
* 基础是指数加权平均
* 核心公式：$$v_t = \beta v_{t-1}+(1-\beta)\theta_t$$

![exponentially_weighted_averages.jpeg](https://i.loli.net/2019/08/20/nKDIaP7S9jkNCqT.jpg)

### 参考

* [第二周：优化算法](http://www.ai-start.com/dl2017/html/lesson2-week2.html)

---




