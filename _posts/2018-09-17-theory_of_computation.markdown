---
layout: post
category: "machinelearning"
title:  "计算学习理论"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

### 基础知识

* 计算学习理论：computational learning theory
* 关于通过”计算“来进行学习的理论，即关于机器学习的理论基础
* 目的：分析学习任务的困难本质，为学习算法提供理论保证，指导算法设计

泛化误差与经验误差：

* 样本集：$$D={(x_1,y_1),...,(x_m,y_m)}, x_i\in\chi$$
* 二分类问题：$$y_i\in Y=\{-1,+1\}$$
* 样本：独立同分布采样获得
* h：$$\chi$$到$$Y$$的一个映射，
* 泛化误差：$$E(h:D)=P_{x \sim D}{(h(x)\ne y)}$$
* 经验误差：$$\hat{E}(h;D)=\frac{1}{m}\sum_{i=1}^m\Pi(h(x_i)\ne y_i)$$
* 独立同分布：经验误差的期望等于泛化误差
* 误差参数：$$\epsilon$$，预先设定的学得模型所应满足的误差要求

---

### PAC学习

* PAC：**P**robably **A**pproximately **C**orrect，概率近似正确学习理论
* 概念：concept，用$$c$$表示，是从样本空间到标记空间的映射，决定了一个样本的标记。若对所有样本，满足$$c(x)=y$$，则称$$c$$为目标概念。【能在所有的样本上预测正确，当然是想要的模型，这里的c可以理解为一个具体的模型？】
* 概念类：$$C$$，希望所学得的目标概念所构成的集合
* 假设空间：$$H$$，对于给定的学习算法，它所考虑的所有可能概念（模型）的集合。学习算法不知道概念类的真实存在，所以通常$$H,C$$是不同的。学习算法会把自认为可能的目标概念集组合构成$$H$$。由于不确定某个概念h是否真是目标概念，所以称为假设（hypothesis）。
* 算法可分：若目标概念$$c\in H$$，则H中存在假设能将所有样本按照与真实标记一致的方向完全分开
* 算法不可分：若目标概念$$c not in H$$，则H中不存在任何假设能将所有样本完全正确分开

* 希望：算法所学得假设模型h尽可能接近目标概念c。
* 不可能完全一致：很多制约因素，采样偶然性等
* 希望以比较大的把握学得比较好的模型
* 概率近似：以较大的概率学得误差满足预设上限的模型

* PAC辨识：PAC identify，对$$0<\epsilon, \delta<1$$，所有$$c\in C$$和分布D，若存在学习算法，其输出假设h满足：$$P(E(h)\leq\epsilon)\ge1-\delta$$，则学习算法：能从假设空间H中PAC辨识概念类C。【在误差$$\epsilon$$允许的范围内，能以较大的概率，学到概念类的近似】
* PAC可学习：PAC learnable，存在一个算法和多项式函数$$poly(.,.,.,.,)$$，使得对于任意的$$m\ge poly(1/\epsilon,1/\delta,size(x),size(c))$$，算法能从假设空间H中PAC辨识概念类C，则称概念类C对假设空间H而言是PAC可学习的。
* PAC学习算法：PAC learning algorithm，若学习算法使得概念类C为PAC可学习的，且算法的运行时间也是多项式函数$$poly(1/\epsilon,1/\delta,size(x),size(c))$$，则称概念类C是高效PAC可学习的。
* 样本复杂度：sample complexity，满足PAC学习算法所需的$$m\ge poly(1/\epsilon,1/\delta,size(x),size(c))$$中的最小的m，称为学习算法的样本复杂度。

* PAC抽象地刻画了机器学习能力的框架
* 什么样的条件下可学得较好的模型？
* 某算法在什么样的条件下可进行有效的学习？
* 需要多少训练样本才能获得较好的模型？

PAC学习中的H的复杂度

*  PCA关键：假设空间H的复杂度
*  H包含所有可能输出的假设，若H=C，则是恰PAC学习的
*  通常H、C不相等，一般来说，H越大，包含任意目标概念的可能性越大，找到具体的目标概念难度也越大
*  H有限时，称H为有限假设空间
*  H无限时，称H为无限假设空间

---

### 有限假设空间

#### 可分情形

* 可分：目标概念c属于假设空间H
* 对于数据集D，如何找出满足误差参数的假设？
* 策略：
	* D中的样本都是由目标概念c赋予的，则D上错误标记的假设肯定不属于目标概念c
	* 可保留与D一致的假设，剔除与D不一致的假设
	* 若D足够大，借助D的样本不断剔除不一致的假设，直到H中仅剩下一个，即为目标概念c

需要多少样本？

* 不加推导的给出(具体推导可参考原文)：[![20190728192101](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190728192101.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190728192101.png)
* 其中H是假设空间，$$\epsilon$$是允许的误差，$$1-\delta$$是近似的概率（$$\delta$$不近似的概率？）

---

#### 不可分情形

这里没看懂，先空着吧！

---

### VC维

* 现实问题：通常是无线假设空间
* 实数域的所有区间、$$R^d$$空间中的所有线性超平面
* 度量假设空间的复杂度：
	* 常见方案：考虑假设空间的VC维

增长函数：
	
* growth function
* 给定假设空间H和数据集$${x_1,x_2,...,x_m}$$，H中的每个假设h都能对样本进行标记（预测），标记结果是：$$h\|D={h(x_1),h(x_2),...,h(x_m)}$$，随着m的增大，H中的所有假设的标记可能结果数目也增大。![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190728235131.png)
* **增长函数：假设空间H对m个样本所能赋予的标记的最大可能结果数。**
* 标记的最大可能结果数越多，H的表示能力越强，**反应假设空间的复杂度**。

对分：

* dichotomy
* m个样本，最多有2^m个可能结果（二分类问题）
* H中的假设对D中样本赋予标记的每种可能结果称为对D的一种”对分“


打散：

* shattering
* 若假设空间H能实现数据集D行的所有对分，即2^m，则称样本集D能被假设空间H打散。

VC维：![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190728235909.png)

* VC(H)=d：表明存在大小为d的样本集能被假设空间H打散，但不是说所有样本集大小为d的样本集都能被打散
* VC的定义与数据分布D无关，只与样本集的大小有关
* 数据分布未知时仍然能够计算样假设空间的VC维
* 计算方法：
	* **若存在大小为d的数据集能被H打散，但是不存在任何大小为d+1的样本集被H打散，则H的VC维是d。**
* 示例：![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190729000518.png)
* VC维与增长函数有密切关系
* 基于VC维的泛化误差界是分布无关、数据独立的

---

### Rademacher复杂度

* VC维的泛化误差界：分布无关、数据独立，普适性强，但是得到的泛化误差界通常较松
* Rademacher复杂度：Rademancher complexity，另一种刻画假设空间复杂度的途径。一定程度考虑了数据分布。具体参考书籍。

---

### 稳定性

* VC维、Rademacher复杂度：与具体学习算法无关，所有学习算法都适用
* 若希望获得与算法有关的结果：
	* 稳定性：考察的是算法在输入发生变化时，输出是否发生较大的变化 [![20190731152415](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190731152415.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190731152415.png)

---

### 参考

* 机器学习周志华第12章
* [The VC dimension @林轩田by红色石头](https://redstonewill.com/222/)





