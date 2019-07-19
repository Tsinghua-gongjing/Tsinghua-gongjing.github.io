---
layout: post
category: "machinelearning"
title:  "sklearn: 官方例子"
tags: [machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

### Miscellaneous

---

#### Compact estimator representations [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/plot_changed_only_pprint_parameter.ipynb)

- 目标：打印模型及参数，可以设置只打印非默认参数值的参数，否则会把全部的参数及其值打印出来。在`0.21.`及之后的版本采用，之前的版本没有。
- 数据集：无
- 模型：无

---

#### Isotonic Regression [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/plot_isotonic_regression.ipynb)

- 目标：在生成数据集中，用线性回归和保序回归对数据进行拟合。保序回归不假设其预测是线性的，所以可以预测非线性的关系。
- 数据集：随机生成的二维数据点
- 模型：Isotonic regression（保序回归），线性回归

---

####  Face completion with a multi-output estimators [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/plot_multioutput_face_completion.ipynb)

- 目标：预测脸图像的下半部分。根据上半部分的像素数据进行模型训练，预测下半部分图像的像素值。图像的下半部分是很多的像素矩阵，所以预测的输出值是一个矩阵，属于multi-output的
- 数据集：fetch_olivetti_faces
- 模型：ExtraTreesRegressor，KNeighborsRegressor，LinearRegression，RidgeCV

---

#### Multilabel classification [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/plot_multilabel.ipynb)

- 目标：对于多label的数据集先进行降维（PCA非监督降维，CCA是监督的降维），再进行分类，多标签的可以训练多个分类器。
- 数据集：随机产生
- 模型：PCA、CCA用于降维，OneVsRestClassifier的两个SVC线性内核的分类器

---

#### Comparing anomaly detection algorithms for outlier detection on toy datasets [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/plot_anomaly_comparison.ipynb)

- 目标：评估不同的异常检测的算法
- 数据集：随机产生的二维数据，从单一的到多modal的分布
- 模型：robust covariance，one-class SVM，isolation forest，local outliner factor

---

#### The Johnson-Lindenstrauss bound for embedding with random projections

- 目标：
- 数据集：
- 模型：

---

#### Comparison of kernel ridge regression and SVR [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/plot_kernel_ridge_regression.ipynb)

- 目标：比较在随机数据上，核岭回归和SVR的效果
- 数据集：随机产生
- 模型：kernel ridge regression（KRR，核脊回归）vs SVR
- KRR：Ridge Regression(RR，脊回归)的kernel版本，与Support Vector Regression(SVR，支持向量回归)类似。引入kernel的RR，也就是KRR，能够处理非线性数据，即，将数据映射到某一个核空间，使得数据在这个核空间上线性可分。
- SVR：支持向量回归

---

#### Explicit feature map approximation for RBF kernels

- 目标：
- 数据集：
- 模型：

---

### Examples based on real world datasets

---

#### Outlier detection on a real data set

- 目标：
- 数据集：
- 模型：

---

#### Compressive sensing: tomography reconstruction with L1 prior (Lasso)

- 目标：
- 数据集：
- 模型：

---

#### Topic extraction with Non-negative Matrix Factorization and Latent Dirichlet Allocation

- 目标：
- 数据集：
- 模型：

---

### Datasets

---

#### The Digit Dataset [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/datasets/plot_digits_last_image.ipynb)

原始的数据集在[UCI machine learning reposity](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits)，包含10992个样本，sklearn这个版本是只去了其中的部分样本：

- 样本数目：1797
- 样本类别：0-9的手写数字图片
- 特征数目：64（8x8）

- 目标：选取了其中的一个数字例子，画了出来。
- 数据集：Handwritten Digits
- 模型：无

---

#### The Iris Dataset [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/datasets/plot_iris_dataset.ipynb)

[维基百科](https://zh.wikipedia.org/wiki/%E5%AE%89%E5%BE%B7%E6%A3%AE%E9%B8%A2%E5%B0%BE%E8%8A%B1%E5%8D%89%E6%95%B0%E6%8D%AE%E9%9B%86)也关于鸢(yuān)尾花卉数据集（Anderson's Iris data set）有详细的介绍：

- 样本数目：150
- 样本类别：3类，鸢(yuān)尾花卉的属，每一类50个数据。分别是山鸢尾（Setosa）、变色鸢尾（Versicolour）和维吉尼亚鸢尾（Virginica）
- 特征数目：4。花萼长度（Sepal.Length），花萼宽度（Sepal.Width），花瓣长度（ Petal.Length），花瓣宽度（Petal.Width），单位：厘米
- 用途：预测花卉属于哪一类


- 目标：展示Iris数据集两个特征的分布餐点图，应用PCA做了数据的降维展示，原本有4个特征，降到3维进行展示（降维后的数据）。
- 数据集：Iris数据集
- 模型：PCA

---

#### Plot randomly generated classification dataset [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/datasets/plot_random_dataset.ipynb)

- 目标：在sklearn中产生用于分类的数据集，主要使用的是模块`sklearn.datasets`，演示的函数包括：`make_classification`，`make_blobs`，`make_gaussian_quantiles`
- 数据集：随机产生
- 模型：无

---

#### Plot randomly generated multilabel dataset [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/datasets/plot_random_multilabel_dataset.ipynb)

- 目标：使用模块`sklearn.datasets`中的函数`make_multilabel_classification`生成多标签的数据。
- 数据集：随机生成
- 模型：无

---


### 参考

* [website: examples @ sklearn](https://scikit-learn.org/stable/auto_examples/index.html)
* [my running notebooks @github](https://github.com/Tsinghua-gongjing/sklearn/tree/master/auto_examples_jupyter)