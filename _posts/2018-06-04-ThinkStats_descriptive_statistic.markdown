---
layout: post
category: "read"
title:  "Think Stats: descriptive statistics?"
tags: [reading, statistics]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

描述性统计量：

1. 均值（mean）：值的总和除以值的数量；平均值（average）：若干种可以用于描述样本的典型值或**集中趋势（central tendency）**的汇总统计量之一。注意根据样本的范围选择合适的描述量。
2. 方差：描述分散情况。方差（variance）的计算公式：$$\begin{align}\sigma^2 = \frac{1}{n}\sum_i(X_i-\mu)^2\end{align}$$，其中$$\begin{align}X_i-\mu\end{align}$$叫做离均差（deviation from the mean），方差的平方根σ叫做标准差（Standard Deviation）。
3. 分布（distribution）：数据值出现的频繁程度。展示方法：直方图（histogram），展示频数或者概率（频数除以总数，归一化normalization的过程）。归一化之后的直方图称为PMF（Probability Mass Function，概率质量函数）：值到概率的映射。
4. 绘制PMF：直方图：利于展示众数、形状、异常值，数目较少时；折现图：数目较多。数目是指取值。
5. 均值和方差也可以通过概率质量函数计算获得。均值：$$\begin{align} \mu = \sum_ip_ix_i \end{align}$$,方差：$$\begin{align} \sigma^2 = \sum_ip_i(x_i-\mu)^2 \end{align}$$
6. 异常值（outliner）：数据采集或处理的错误，罕见结果，需要去除掉（trim）。
7. 相对风险（relative risk）：代表两个概率的比值。
8. 条件概率（conditional probability）：依赖于某个条件的概率。限定了某些条件（已知了这些），对应事件的概率会发生改变。
9. 汇报结果：如何汇报结果还取决于具体目的。如果是要证明某种影响的显著性，可以选择汇总统计量，如强调差异的相对风险。如果是要说服某个患者，则可以选择能反映特定情况下差异的统计量。

术语：

* 区间（bin） 将相近数值进行分组的范围。
* 集中趋势（central tendency） 样本或总体的一种特征，直观来说就是最能代表平均水平的值。
* 临床上有重要意义（clinically significant） 分组间差异等跟实践操作有关的结果。
* 条件概率（conditional probability） 某些条件成立的情况下计算出的概率。
* 分布（distribution） 对样本中的各个值及其频数或概率的总结。
* 频数（frequency） 样本中某个值的出现次数。
* 直方图（histogram） 从值到频数的映射，或者表示这种映射关系的图形。
* 众数（mode） 样本中频数最高的值。
* 归一化（normalization） 将频数除以样本大小得到概率的过程。
* 异常值（outlier） 远离集中趋势的值。
* 概率（probability） 频数除以样本大小即得到概率。
* 概率质量函数（Probability Mass Function，PMF） 以函数的形式表示分布，该函数将值映射到概率。
* 相对风险（relative risk） 两个概率的比值，通常用于衡量两个分布的差异。
* 分散（spread） 样本或总体的特征，直观来说就是数据的变动有多大。
* 标准差（standard deviation） 方差的平方根，也是分散的一种度量。
* 修剪（trim） 删除数据集中的异常值。
* 方差（variance） 用于量化分散程度的汇总统计量。