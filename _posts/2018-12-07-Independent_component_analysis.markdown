---
layout: post
category: "machinelearning"
title:  "ICA独立成分分析"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## 概述

独立成分分析（Independent Component Analysis， ICA）：类似于PCA，找到一组新的基向量（basis）来表征样本数据，但是PCA用于服从高斯分布的数据，ICA用于服从非高斯分布的数据。

经典案例就是“鸡尾酒宴会问题”：

> 在一个聚会场合中，有$$n$$个人同时说话，而屋子里的任意一个话筒录制到底都只是叠加在一起的这$$n$$个人的声音。但如果假设我们也有 $$n$$个不同的话筒安装在屋子里，并且这些话筒与每个说话人的距离都各自不同，那么录下来的也就是不同的组合形式的所有人的声音叠加。使用这样布置的$$n$$个话筒来录音，能不能区分开原始的$$n$$个说话者每个人的声音信号呢？

![](https://danieljyc.github.io/img/1402665355695.png)

## 知识点

1、适用场景：
  - 数据不服从高斯分布
  - 数据信号是相互独立的

2、鸡尾酒问题中的ICA算法：[![ICA.png](https://i.loli.net/2019/06/10/5cfe1475d694856583.png)](https://i.loli.net/2019/06/10/5cfe1475d694856583.png)

3、ICA vs PCA：
   - ICA：样本数据由独立非高斯分布的隐含因子产生，隐含因子个数等于特征数，更适合用来还原信号
   - PCA：K个正交的特征，更适合降维 [![ICA_vs_PCA.png](https://i.loli.net/2019/06/10/5cfe1958d4f5597241.png)](https://i.loli.net/2019/06/10/5cfe1958d4f5597241.png)


## 参考

* [机器学习15-3—独立成分分析ICA（Independent Component Analysis）](https://danieljyc.github.io/2014/06/13/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A015-3--%E7%8B%AC%E7%AB%8B%E6%88%90%E5%88%86%E5%88%86%E6%9E%90ica%EF%BC%88independent-component-analysis%EF%BC%89/)
* [独立成分分析（Independent Component Analysis）](https://www.cnblogs.com/jerrylead/archive/2011/04/19/2021071.html)
* [CS229 课程讲义中文翻译: 第十二部分 独立成分分析](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes11)
* [ICA @UFLDL Tutorial](http://ufldl.stanford.edu/tutorial/unsupervised/ICA/)