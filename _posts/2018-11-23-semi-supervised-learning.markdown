---
layout: post
category: "machinelearning"
title:  "半监督学习"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

### 标记样本、未标记样本

* 标记样本：labeled，给出了y值的训练样本，$$D_l=\{(x_1,y_1),...,(x_l,y_l)\}$$
* 未标记样本：unlabeled，类别标记未知的样本，$$D_u=\{(x_{l+1},y_{l+1}),...,(x_{l+u},y_{l+u})\}$$
* 传统监督学习：仅使用$$D_l$$构建模型，$$D_u$$包含信息被浪费
* 若$$D_l$$较小，训练样本不足，模型泛化能力不佳
* 问题：能否在构建模型的过程中将$$D_u$$利用起来？

---

### 主动学习 vs 半监督

主动学习：active learning

* 引入额外的专家知识，通过与外界的交互来将部分未标记样本转变为有标记样本
* 对于未标记样本，拿来一个，询问专家是好还是坏样本，加入有标记样本数据集进行训练，如果添加的样本能增强模型能力，也可降低标记成本

未标记样本的数据分布可提供信息

* 标记和未标记样本均：独立同分布
* 未标记样本数据多，数据分布更接近于真实分布
* 可帮助判断
* 比如可帮助判断分类：[![20190801215112](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190801215112.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190801215112.png)

半监督学习：

* 学习器不依赖外界交互、自动的利用未标记样本提升学习性能
* 显示需求强烈，获取标记很费力
* 利用未标记样本需要一些假设：
	* 聚类假设：最常见，假设数据存在簇结构，同一个簇的样本属于同一个类别
	* 流形假设：常见，假设数据分布在一个流形结构上，邻近的样本拥有相似的输出值。邻近用相似程度刻画，因此可看做聚类的推广。
	* 假设本质：相似的样本拥有相似的输出

分类：

* 纯半监督学习：pure，训练数据中的为标记样本不是待预测的数据。更普世、泛化。
* 直推学习：transductive learning，学习过程中所考虑的为标记样本恰是待预测数据 [![20190801215826](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190801215826.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190801215826.png)

---

### 生成式方法

* 基于生成式模型
* 假设所有数据（标记+未标记）由同一潜在模型生成
* 通过潜在模型的参数将未标记数据与学习目标联系起来
* 未标记数据的标记：模型的确实参数
* 可基于EM算法进行极大似然估计
* 不同的**潜在模型**（人为经验等进行的），对应不同的学习器

---

### 参考

* 机器学习周志华第13章





