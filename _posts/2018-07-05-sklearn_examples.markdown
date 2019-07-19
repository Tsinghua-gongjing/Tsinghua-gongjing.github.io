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
- 数据集：fetch_20newsgroups_vectorized
- 模型：

---

#### Comparison of kernel ridge regression and SVR [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/plot_kernel_ridge_regression.ipynb)

- 目标：比较在随机数据上，核岭回归和SVR的效果
- 数据集：随机产生
- 模型：kernel ridge regression（KRR，核脊回归）vs SVR
- KRR：Ridge Regression(RR，脊回归)的kernel版本，与Support Vector Regression(SVR，支持向量回归)类似。引入kernel的RR，也就是KRR，能够处理非线性数据，即，将数据映射到某一个核空间，使得数据在这个核空间上线性可分。
- SVR：支持向量回归

---

#### Explicit feature map approximation for RBF kernels [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/plot_kernel_approximation.ipynb)

- 目标：通过不同的特征映射(变换)近似高斯内核。
- 数据集：digitals
- 模型：内核近似函数`RBFSampler`, `Nystroem`

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

### Classification

---

#### Recognizing hand-written digits [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/classification/plot_digits_classification.ipynb)

- 目标：预测手写数字，前一半数据用于训练，预测后一半数据，画出了几个例子
- 数据集：digital数据集
- 模型：SVM高斯内核

- 图像数据转换：`data = digits.images.reshape((n_samples, -1))`，把原来的(1797, 8, 8)数据转换为(1797, 64)维，进行训练
- 使用了评估量：classification_report+confusion_matrix

---

### Clustering

---

#### Feature agglomeration [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/cluster/plot_digits_agglomeration.ipynb)

- 目标：对于手写数字数据集，合并相似的特征，进行特征聚集。这里是将原始的数据集经过特征聚集从64变为32（reduced），基于这个数据也可以重建回原来的数据集（restored）。
- 数据集：digitals
- 模型：特征聚集`cluster.FeatureAgglomeration`

---

#### Demo of DBSCAN clustering algorithm [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/classification/plot_dbscan.ipynb)

- 目标：在生成的数据集上进行DBSCAN聚类，并计算不同的评估聚类效果的统计量
- 数据集：生成数据集，750x2
- 模型：DBSCAN聚类

---
#### Color Quantization using K-Means [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/cluster/plot_color_quantization.ipynb)

- 目标：对一个RGB图片（427x640x3），用kmeans对颜色进行聚类，得到64个新的颜色中心值，然后预测原来图片的每个像素点的新的RGB值（从这64个选取一个），然后就可以画新的只有64个颜色值所表示的图片了。实现了用64个颜色表示原来的96615个颜色的，可以极大的降低了图片存储的大小。
- 数据集：一张图片（china.jpg），427x640x3
- 模型：kmeans聚类

---

### Ensemble methods

---

#### Discrete versus Real AdaBoost [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Multi-class AdaBoosted Decision Trees [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Decision Tree Regression with AdaBoost [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Two-class AdaBoost [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

###============================================================ [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### OOB Errors for Random Forests [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Feature transformations with ensembles of trees [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Feature importances with forests of trees [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Pixel importances with a parallel forest of trees [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Plot the decision surfaces of ensembles of trees on the iris dataset [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Early stopping of Gradient Boosting [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Gradient Boosting Out-of-Bag estimates [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Prediction Intervals for Gradient Boosting Regression [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Gradient Boosting regression [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Gradient Boosting regularization [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### IsolationForest example [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Hashing feature transformation using Totally Random Trees [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Comparing random forests and the multi-output meta estimator [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Plot the decision boundaries of a VotingClassifier [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Plot class probabilities calculated by the VotingClassifier [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

#### Plot individual and voting regression predictions [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/ensemble/)

- 目标：
- 数据集：
- 模型：

---

### Tutorial exercises

---

#### Digits Classification Exercise [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/exercises/plot_digits_classification_exercise.ipynb)

- 目标：对digital数据进行分类预测，前90%数据集训练，后10%用于测试
- 数据集：digital数据集
- 模型：KNN，logistic regression

---

#### Cross-validation on Digits Dataset Exercise [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/exercises/plot_cv_digits.ipynb)

- 目标：对digital数据使用SVM进行分类预测，并进行cross validation，所以这里不像上面的那样，自行拆分训练集和测试集。同时，还测试了模型SVC中的参数`C`(Penalty parameter C of the error term.)取不同的数值大小时，随对应的`cross_val_score`。
- 数据集：digital数据集
- 模型：SVM

---

#### SVM Exercise [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/exercises/plot_iris_exercise.ipynb)

- 目标：对Iris数据进行分类预测，使用SVM模型，并尝试不同的内核（参数`kernel`指定）比较效果。
- 数据集：Iris数据集
- 模型：SVM的3个不同内核函数（linear线性，rbf高斯（默认），poly多项式）

---

#### Cross-validation on diabetes Dataset Exercise [[notebook]](https://github.com/Tsinghua-gongjing/sklearn/blob/master/auto_examples_jupyter/exercises/plot_cv_diabetes.ipynb)

- 目标：对diabetes数据进行多分类预测，使用交叉检验
- 数据集：diabetes数据集
- 模型：线性模型LassoCV和Lasso

- 数据集[diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
- 样本数目：442
- 特征数目：10，-.2 < x < .2之间的实数
- 类别数目：25 - 346，整数，所以是多分类的，不是二分类的

---

### 参考

* [website: examples @ sklearn](https://scikit-learn.org/stable/auto_examples/index.html)
* [my running notebooks @github](https://github.com/Tsinghua-gongjing/sklearn/tree/master/auto_examples_jupyter)