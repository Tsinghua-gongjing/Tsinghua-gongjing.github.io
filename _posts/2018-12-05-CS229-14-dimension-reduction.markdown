---
layout: post
category: "machinelearning"
title:  "[CS229] 14: Dimensionality Reduction"
tags: [python, machine learning]
---

## 14: Dimensionality Reduction

1. 降维：非监督学习算法，用更简化的表示方法表征收集的数据。
2. 为啥降维 -》**压缩**：
   - 高维的数据用低维表征
   - 加速算法，节省存储空间
   - 例子：1）二维数据点用一维表示，x1和x2只是两个不同的单位（冗余）；2）三维数据点用二维表示，所有的点在一个平面上，此平面用二维即可定义：[![PCA_compression.jpeg](https://i.loli.net/2019/05/19/5ce1746bce41892268.jpeg)](https://i.loli.net/2019/05/19/5ce1746bce41892268.jpeg)
   - 实际中可以做到：1000维 =》100维
3. 为啥降维 -》**可视化**：
   - 高维数据难以可视化
   - 降维可以将数据以人可以理解的方式展现出来
   - 比如一个表格，使用了50多个特征描述不同的国家，从这里我们能得到什么信息？
   - 通常很难理解具体的降维之后的特征所表示的意思
4. 主成分分析（PCA）：
   - 例子：2D数据降至1D，所以需要找一个一维的向量，使得所有点在这个向量的投射误差最小：
   - 投映误差（projection error）：点到向量的垂直距离（orthogonal distance）![](http://www.holehouse.org/mlclass/14_Dimensionality_Reduction_files/Image%20[8].png)
   - 在做PCA之前，一定要对数据进行预处理：**平均值归一化**（mean normalization）和**特征缩放**（feature scaling）
   - 广泛定义：
   - nD =》kD，在将数据从D维降至k维时，需要找k个向量，使得所有数据投射到这些向量时，投映误差最小。3D -》2D：找两个向量。
5. PCA vs 线性回归：
   - PCA和线性回归很像，但PCA不是线性回归，其差别很大
   - 线性回归：拟合直线，使得**y方向的差值（vertical distance，预测的误差）**最小，因为线性回归是有y值的；
   - PCA：没有y值，只有特征值（且这些特征值都是同等对待的）。所以是找向量，误差的计算用的是**垂直距离（orthogonal distance）**。
6. PCA算法：
   - 【预处理：均值归一化】：对于每个特征，进行均值归一化，x =》 x-mean(x)。每个特征的均值归一化到0.
   - 【预处理：特征缩放】：当特征的区间差异很大时，需要进行缩放使得数值在可比较的区间里。1）Biggest - smallest；2）Standard deviation（标准差缩放更常见）
   - **计算2个向量：1）新的平面向量u；2）新的特征向量z。**
   - 比如2D -》1D，要计算向量u和新的特征向量：![](http://www.holehouse.org/mlclass/14_Dimensionality_Reduction_files/Image%20[11].png)
7. 算法描述（[这里](https://www.cnblogs.com/xingshansi/p/6445625.html)有计算的步骤描述，可参考）：
   - 1）数据预处理：中心化，归一化
   - 2）根据样本计算协方差矩阵
   - 3）利用特征值/奇异值分解，求特征值和特征向量
   - 4）用特征向量构造投影矩阵
   - 5）利用投影矩阵和样本数据，计算降维的数据
8. 如何选取k：
   - 通常会定义一个**比值=平均投影误差/数据本身总方差**：![](http://www.holehouse.org/mlclass/14_Dimensionality_Reduction_files/Image%20[19].png)
   - 这个数值越小越好，越小说明丢失的信息越少，保留的信息越多。
   - 实际操作：从k=1开始尝试，看哪个能满足上面的比值<=0.01：![](http://www.holehouse.org/mlclass/14_Dimensionality_Reduction_files/Image%20[20].png)
9. PCA可加速监督式学习算法：
   - 比如对于具有超高维特征的数据，可先用PCA进行降维，使用降维后的数据和原始的标签进行模型的训练，再去对于新的数据进行预测。
   - 新数据：怎么预测？特征数目或者叫维度不一样！！！在上面的PCA中计算了参数投影矩阵。
10. PCA应用：
   - 压缩：减少存储，加速算法。保留多少的variance来确定k值。
   - 可视化：通常降至2-3维
   - 错误使用PCA避免过拟合：以为具有更少的特征就能避免过拟合，实则不然。应该使用正则化防止过拟合。
   - 至少95%-99%的信息是需要保留的。
   - 在使用PCA之前，先尝试不使用时能否达到预期的效果。