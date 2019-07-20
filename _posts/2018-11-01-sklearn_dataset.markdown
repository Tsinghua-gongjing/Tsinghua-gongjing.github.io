---
layout: post
category: "machinelearning"
title:  "sklearn: 数据集加载"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

#### 1. 数据集

* 类：`sklearn.datasets`
* 通用数据
* 生成数据
* 获取真实数据

#### 2. 通用数据集

三种API接口：

* `loader`：加载小的标准数据集
* `fetchers`：下载大的真实数据集
* `generate functions`：生成受控的合成数据集

#### 3. 通用标准数据集

##### boston house-prices 

[![dataset_boston_house_price.jpeg](https://i.loli.net/2019/07/21/5d333b955114276295.jpeg)](https://i.loli.net/2019/07/21/5d333b955114276295.jpeg)

##### Iris 

[![dataset_iris.jpeg](https://i.loli.net/2019/07/21/5d333e668991227533.jpeg)](https://i.loli.net/2019/07/21/5d333e668991227533.jpeg)

##### diabetes 

[![dataset_diabetes.jpeg](https://i.loli.net/2019/07/21/5d3344b405a5696367.jpeg)](https://i.loli.net/2019/07/21/5d3344b405a5696367.jpeg)

##### digitals 

[![dataset_digitals.jpeg](https://i.loli.net/2019/07/21/5d3344c8e2b1413841.jpeg)](https://i.loli.net/2019/07/21/5d3344c8e2b1413841.jpeg)

##### linnerud 

[![dataset_linnerrud.jpeg](https://i.loli.net/2019/07/21/5d3344d9cce7f72487.jpeg)](https://i.loli.net/2019/07/21/5d3344d9cce7f72487.jpeg)

##### wine 

[![dataset_wine.jpeg](https://i.loli.net/2019/07/21/5d334545274a919488.jpeg)](https://i.loli.net/2019/07/21/5d334545274a919488.jpeg)

##### breast cancer 

[![dataset_breast_cancer.jpeg](https://i.loli.net/2019/07/21/5d3344ed9352361529.jpeg)](https://i.loli.net/2019/07/21/5d3344ed9352361529.jpeg)

### 参考

* [据集加载工具@sklearn 中文](https://sklearn.apachecn.org/#/docs/47?id=_6-%E6%95%B0%E6%8D%AE%E9%9B%86%E5%8A%A0%E8%BD%BD%E5%B7%A5%E5%85%B7)