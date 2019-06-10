---
layout: post
category: "genomics"
title:  "进化树"
tags: [genomics]
---

## 进化树

### 定义

* 系统发生树又称演化树或进化树，是表明被认为具有共同祖先的各物种间演化关系的树状图。是一种亲缘分支分类方法（cladogram）。在图中，每个节点代表其各分支的最近共同祖先，而节点间的线段长度对应演化距离（如估计的演化时间）。

### 建树步骤

* 1、多序列比对
* 2、选择建树算法构建进化树
* 3、进化树构建结果的评估

### 多序列比对软件

* clustwal。也可用命令行形式：`clustalo -i input.fa -o input.align.fa`
* MEGA

### 构建算法

* 近邻结合法 neighbor-joining (NJ)
* 最大简约法 maximum parsimony (MP)
* 最大似然估计 maximum likelihood (ML)
* 贝叶斯法 Bayesian

### 软件

* Phylip、MrBayes、Simple Phylogeny

### 进化树含义

[这里](http://www.360doc.com/content/17/1120/01/33459258_705424795.shtml)有两张图解释了进化树的各含义，可以参考一下。

主要的几点如下：

* 根节点：所有物种（序列）的共同祖先（下图中酒红色的点）
* 节点：分类单元（下图中蓝色的点和绿色的点）
* 进化支：>=2个物种（序列）及其祖先组成的分支
* 距离标尺：进化树的比例尺，序列的差异度的大小，比如下图是0.07
* 分支长度：节点之间或者节点与祖先之间的累计支长，越小则差异越小
* 自展值：表征可信度的，因为建树的过程需要通过bootstrap方法重复多次，一般小于50%（0.5）的会被认为不可靠

![](https://image.slidesharecdn.com/basicconceptsinsystamaticstaxonomyandphylogenetictree-180322170316/95/basic-concepts-in-systamaticstaxonomy-and-phylogenetic-tree-28-638.jpg?cb=1539454564)

## 参考

* [一文读懂进化树](http://www.360doc.com/content/17/1120/01/33459258_705424795.shtml)
* [Phylogenetic trees @ NCBI](https://www.ncbi.nlm.nih.gov/CBBresearch/Przytycka/download/lectures/PCB_Lect11_Phylogen_Trees.pdf)