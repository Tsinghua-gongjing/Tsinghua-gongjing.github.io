---
layout: post
category: "machinelearning"
title:  "随机森林"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

### 随机森林算法的基础

* 是bagging和决策树的结合
* bagging：通过bootstrap的方式，从原始数据集D得到新的数据集D‘，然后对每个得到的数据集，使用base算法得到相应的模型，最后通过投票形式组合成一个模型G，即最终模型。
* 决策树：通过递归，利用分支条件，将原始数据集D进行切割，成为一个个的子树结构，到终止形成亦可完整的树形结构。最终的模型G是相应的分支条件和分支树递归组成。

* bagging：减少不同数据集对应模型的方差。因为是投票机制，所以有平均的功效。
* 决策树：增大不同的分支模型的方差。数据集进行切割，分支包含样本数在减少，所以不同的分支模型对不太的数据集会比较敏感，得到较大的方差。
* 结合两者：随机森林算法，将完全长成的CART决策树通过bagging的形式结合起来，得到一个庞大的决策模型。
* **random forest(RF)=bagging+fully-grown CART decision tree**

---

### 实现

### Python源码版本

### sklearn版本

---

### 参考

* [Random Forest](https://github.com/apachecn/ntu-hsuantienlin-ml/blob/master/28.md)





