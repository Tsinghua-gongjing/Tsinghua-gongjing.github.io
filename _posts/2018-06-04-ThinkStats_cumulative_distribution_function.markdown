---
layout: post
category: "read"
title:  "Think Stats: cumulative disribution function?"
tags: [reading, statistics]
---

累积分布函数：

1. PMF缺点：如果要处理的数据比较少，PMF很合适。但随着数据的增加，每个值的概率就会降低，而随机噪声的影响就会增大。解决策略：1）根据bin划分区间，如何确定bin的数目比较难；2）累计分布函数（Cumulative Distribution Function，CDF）。
2. 百分位数（percentile）：不高于某个值所占的比例再乘以100。转换：给定值，计算百分位数；对于给定的百分位数，计算对应的值。
3. CDF：值到其在分布中百分等级的映射。如果x比样本中最小的值还要小，那么CDF(x)就等于0。如果x比样本中的最大值还要大，那么CDF(x)就是1。
4. 条件分布：根据某个条件选择的数据子集的分布。通常不同的实验，条件不同，不能直接的相互比较。可以通过转换为对应组别的百分位数进行比较。
5. 再抽样（resampling）：根据已有的样本生成随机样本的过程。有放回和无放回：[取球问题](http://wikipedia.org/wiki/Urn_problem)。
6. CDF推出：中位数（median）就是百分等级是50的值；25和75百分等级通常用来检查分布是否对称，这两者间的差异称为四分差（interquartile range），表示分布的分散情况。

术语：

* 条件分布 （conditional distribution） 在满足一定前提条件下计算出的分布。
* 累积分布函数（Cumulative Distribution Function，CDF） 将值映射到其百分等级的函数。
* 四分差 （interquartile range） 表示总体分散情况的值，等于75和25百分等级之间的差。
* 百分位（percentile） 与百分等级相关联的数值。
* 百分等级 （percentile rank） 分布中小于或等于给定值的值在全部值中所占的百分比。
* 放回 （replacement） 在抽样过程中，“有放回”表示对于每次抽样，总体都是不变的。“无放回”表示每个元素只能选择一次。
* 再抽样 （resampling） 根据由样本计算得到的分布重新生成新的随机样本的过程。