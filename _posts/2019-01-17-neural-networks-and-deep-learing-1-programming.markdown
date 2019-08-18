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

### 逻辑回归的代价函数

* 损失函数L：
	* 衡量预测输出值与实际值有多接近
	* 一般使用平方差或者平方差的一半
* 逻辑回归损失函数loss function：
	* 逻辑回归如果也使用平方差：
		* 优化目标不是凸优化的
		* 只能找到多个局部最优解
		* 梯度下降法很可能找不到全局最优
	* 逻辑回归使用对数损失：$$L(\hat y, y)=-ylog(\hat y)-(1-y)log(1-\hat y)$$
	* 单个样本的损失
	* **为啥是这个损失函数？【直观理解】**
	* 当y=1时，损失函数$$L=-log(\hat y)$$，L小，则$$\hat y$$尽可能大，其取值本身在[0，1]之间，所以会接近于1（也接近于此时的y）
	* 当y=0时，损失函数$$L=-log(1-\hat y)$$，L小，则$$1-\hat y$$尽可能大，$$\hat y$$尽可能小，其取值本身在[0，1]之间，所以会接近于0（也接近于此时的y）
* 代价函数cost function
	* 参数的总代价
	* 对m个样本的损失函数求和然后除以m

---

### 梯度下降法

* 逻辑回归的代价函数是凸函数（convex function），像一个大碗一样，具有一个全局最优。如果是非凸的，则存在多个局部最小值。![convex_vs_nonconvex.jpeg](https://i.loli.net/2019/08/18/EW5bVOoMadyJkYw.jpg)
* 梯度下降：
	* 初始化w和b参数。逻辑回归，无论在哪里初始化，应该达到统一点或大致相同的点。
	* 朝最陡的下坡方向走一步，不断迭代
	* 直到走到全局最优解或者接近全局最优解的地方
* 迭代：
	* 公式：$$w := w - \alpha \frac{dJ(w)}{dw} 或者 w := w - \alpha \frac{\partial J(w,b)}{\partial w}$$
	* 学习速率：$$\alpha$$，learning rate，控制步长，向下走一步的长度
	* 函数J(w)对w求导：$$\frac{dJ(w)}{dw}$$，就是斜率(slope)
	* 偏导：$$\partial$$表示，读作round，$$\frac{\partial J(w,b)}{\partial w}$$就是J(w,b)对w求偏导
	* 在逻辑回归还有参数b，同样的进行迭代更新

	* 一个参数：求**导数**（derivative），用小写字母d表示
	* 两个参数：求**偏导数**（partial derivative），用$$\partial$$表示

---

### 参考

* [第二周：神经网络的编程基础](http://www.ai-start.com/dl2017/html/lesson1-week2.html)

---




