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

#### Compact estimator representations

- 目标：
- 数据集：
- 模型：

---

#### Isotonic Regression

- 目标：
- 数据集：
- 模型：

---

####  Face completion with a multi-output estimators [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/plot_multioutput_face_completion.ipynb)

- 目标：预测脸图像的下半部分。根据上半部分的像素数据进行模型训练，预测下半部分图像的像素值。图像的下半部分是很多的像素矩阵，所以预测的输出值是一个矩阵，属于multi-output的
- 数据集：fetch_olivetti_faces
- 模型：ExtraTreesRegressor，KNeighborsRegressor，LinearRegression，RidgeCV

---

#### Multilabel classification

- 目标：
- 数据集：
- 模型：

---

#### Comparing anomaly detection algorithms for outlier detection on toy datasets

- 目标：
- 数据集：
- 模型：

---

#### The Johnson-Lindenstrauss bound for embedding with random projections

- 目标：
- 数据集：
- 模型：

---

#### Comparison of kernel ridge regression and SVR

- 目标：
- 数据集：
- 模型：

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

#### The Digit Dataset

- 目标：
- 数据集：
- 模型：

---

#### The Iris Dataset [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/datasets/plot_iris_dataset.ipynb)

- 目标：展示Iris数据集两个特征的分布餐点图，应用PCA做了数据的降维展示，原本有4个特征，降到3维进行展示（降维后的数据）。
- 数据集：Iris数据集
- 模型：PCA

---

#### Plot randomly generated classification dataset

- 目标：
- 数据集：
- 模型：

---

#### Plot randomly generated multilabel dataset

- 目标：
- 数据集：
- 模型：

---


### 参考

* [website: examples @ sklearn](https://scikit-learn.org/stable/auto_examples/index.html)
* [my running notebooks @github](https://github.com/Tsinghua-gongjing/sklearn/tree/master/auto_examples_jupyter)