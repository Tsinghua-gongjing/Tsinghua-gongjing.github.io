---
layout: post
category: "machinelearning"
title:  "【1-2】神经网络的编程基础"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

### 二分类

* 神经网络：对于m个样本，通常不用for循环去便利整个训练集
* 神经网络的训练过程：
	* 前向传播：forward propagation
	* 反向传播：backward propagation
* 例子：逻辑回归

* 二分类:
	* 目标：识别图片是猫或者不是
	* 提取特征向量：3通道的特征向量，每一个是64x64的，最后是64x64x3 ![](http://www.ai-start.com/dl2017/images/e173fd42de5f1953deb617623d5087e8.png)

* 符号表示：
	* 训练样本的矩阵表示X：每一个样本是一列，矩阵维度：特征数x样本数
	* 输出标签y：矩阵维度：1x样本数 ![](http://www.ai-start.com/dl2017/images/55345ba411053da11ff843bbb3406369.png)

---

### 逻辑回归

* 二分类：
	* 输入特征向量，判断是不是猫
	* 算法：输出预测值$$\hat y$$对实际值$$y$$的估计
	* 正式的定义：让预测值$$\hat y$$表示实际值$$y$$等于1的可能性
* 例子：
	* 图片识别是否是猫
	* 尝试：$$\hat y=w^Tx+b$$，线性函数，对二分类不是好的算法，因为是想“让预测值$$\hat y$$表示实际值$$y$$等于1的可能性”，所以$$\hat y$$应该在【0，1】之间。
* sigmoid函数：
	* 当z很大时，整体值接近于1（1/1+0）
	* 当z很小时，整体值接近于0（1/1+很大的数）
	* ![](http://www.ai-start.com/dl2017/images/7e304debcca5945a3443d56bcbdd2964.png)
* 让机器学习参数$$w$$及$$b$$，使得$$\hat y$$称为对$$y=1$$这一情况的概率的一个很好的估计。

---

### 参考

* [第二周：神经网络的编程基础](http://www.ai-start.com/dl2017/html/lesson1-week2.html)

---




