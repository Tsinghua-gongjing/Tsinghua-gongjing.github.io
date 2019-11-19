---
layout: post
category: "statistics"
title:  "MSE vs Pearson correlation coefficient"
tags: [reading, statistics]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 模型得到的loss越小，是否预测值和真实值之间的corr就越大？

最近遇到个问题，通过不同的方式训练得到了两个模型。在模型训练时，使用的是验证集数据的loss，挑选最小loss的。在得到模型之后，使用验证集数据集进行预测并比较预测值和真实值之间的相关性。但是出现了一个异常情况是，其中一个模型的loss更小，但是计算出来的相关系数更小？

[![20191119135356](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191119135356.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191119135356.png)

具体解释可以参考这里：[Confusion regarding correlation and mse](https://stats.stackexchange.com/questions/34033/confusion-regarding-correlation-and-mse)

MSE和相关性系数存在关系，但是不仅取决于相关系数，预测值的方差也会影响MSE的值，所以对于两个预测值的集合，如果方差不一样的，那么久不一定是完全的线性关系。

注意：

* 首先：得保证比较的MSE和相关系数是同一个东西，比如这里提到的因为missing value导致的数目不同，不能直接比较。