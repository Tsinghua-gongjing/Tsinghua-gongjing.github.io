---
layout: post
category: "statistics"
title:  "期望、方差、协方差、相关系数"
tags: [reading, statistics]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

---

### 期望

* 定义：实验中每次可能结果的概率乘以其结果的总和
* 反映随机变量平均取值的大小
* 线性运算：$$E(ax+by+c)=aE(x)+bE(y)+c$$
* 推广形式：$$E(\sum_{k=1}^{n} a_i x_i + c) = \sum_{k=1}^{n}(a_i E(x_i)) + c$$
* 函数期望：
	* f(x)是x的函数
	* 离散函数：$$E(f(x))=\sum_{k=1}^{n}f(x_k)P(x_k)$$
	* 连续函数：$$E(f(x))=\int_{-\infty}^{\infty}f(x)p(x)dx$$
* 性质：
	* 函数的期望大于期望的函数，即$$E(f(x)) >= f(E(x))$$
	* 一般情况下，乘积的期望不等于期望的乘积
	* 如果X和Y相互独立，则$$E(xy)=E(x)E(y)$$

---

### 方差

* 度量随机变量和其数学希望之间的偏离程度
* 是一种特殊的期望
* 定义：$$Var(x)=E((x-(E(x))^2))$$
* 性质：
	* 变种：$$Var(x)=E(x^2)-(E(x))^2$$
	* 常数的方差为0
	* 方差不满足线性性质
	* 如果X和Y相互独立，则$$Var(ax+by)=a^2Var(x)+b^2Var(y)$$

```python
import numpy as np 
arr = [1,2,3,4,5,6]
#求均值
arr_mean = np.mean(arr)
#求方差
arr_var = np.var(arr)
#求标准差
arr_std = np.std(arr,ddof=1)
print("平均值为：%f" % arr_mean)
print("方差为：%f" % arr_var)
print("标准差为:%f" % arr_std)
```

---

### 协方差

* 衡量**两个变量**线性相关性强度及变量尺度
* 定义：$$Cov(x,y)=E((x-E(x))(y-E(y)))$$
* 方差是一种特殊的协方差：
	* 当X=Y时，$$Cov(x,y)=Var(x)=Var(y)$$
* 性质：
	* 两个独立变量的协方差为0。因为此时独立的随机变量x、y满足：$$E[xy]=E(x)E(y)$$
	* 计算公式：$$Cov(\sum_{i=1}^{m}a_{i}x_{i}, \sum_{j=1}^{m}b_{i}y_{i})=\sum_{i=1}^{m}\sum_{j=1}^{m}a_{i}b_{j}Cov(x_{i}y_{i})$$
	* 特殊情况：$$Cov(a+bx,c+dy)=bdCov(x,y)$$
* 理解：
	* 表示的是两个变量总体误差的期望
	* 如果两个变量的**趋势一致**，比如变量x大于自身期望且y也大于自身期望，那么两个变量x、y之间的**协方差就是正值**
	* 如果两个变量的**趋势相反**，比如变量x大于自身期望但是y小于自身期望，那么两个变量x、y之间的**协方差就是负值**

```python
from numpy import array
from numpy import cov
x = array([1,2,3,4,5,6,7,8,9])
print(x)
y = array([9,8,7,6,5,4,3,2,1])
print(y)
Sigma = cov(x,y)[0,1]
print(Sigma)

# [1 2 3 4 5 6 7 8 9]
# [9 8 7 6 5 4 3 2 1]

# -7.5
```

---

### 相关系数

* 研究**两个变量**之间线性相关程度的量
* 为什么引入相关系数？
	* 协方差就是描述两个变量X、Y的相关程度的
	* 相同量纲下，协方差没有问题
	* 但是当x、y属于不同量纲时，协方差会在数值上表现出很大的差异
	* 因而引入了相关系数
* 定义：$$Corr(x,y)=\frac{Cov(x,y)}{\sqrt({Var(x)Var(y)})}$$
* 性质：
	* 取值范围在[-1, 1]，可看成无量纲的协方差
	* 值越接近于1，正相关性越强；越接近于-1，负相关性越强；等于0时，没有相关性。

```python
from scipy import stats

# two sample rank test
def sig_spearman_corr(x,y):
    p=stats.spearmanr(x,y)[0]
    return p

def sig_pearson_corr(x,y):
    p=stats.pearsonr(x,y)[0]
    return p
```

---

### 参考

* [期望、方差、协方差及相关系数的原理理解和计算](https://blog.csdn.net/qq_29540745/article/details/52132836)
* [深度学习500问-第一章 数学基础](https://github.com/Tsinghua-gongjing/DeepLearning-500-questions/tree/master/ch01_%E6%95%B0%E5%AD%A6%E5%9F%BA%E7%A1%80)

