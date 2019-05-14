---
layout: post
category: "machinelearning"
title:  "[CS229] 06: Logistic Regression"
tags: [python, machine learning]
---

## 06: Logistic Regression

1. 逻辑回归：分类，比如email是不是垃圾邮件，肿瘤是不是恶性的。预测y值（label），=1（positive class），=0（negative class）。
2. 分类 vs 逻辑回归（逻辑回归转换为分类）：
   - 分类：值为0或1（是离散的，且只能取这两个值）。
   - 逻辑回归：预测值在[0,1之间]。
   - 阈值法：用逻辑回归模型，预测值>=0.5，则y=1，预测值<0.5，则y=0.
3. 逻辑回归函数（假设，hypothesis）：
   - 公式：![](http://www.holehouse.org/mlclass/06_Logistic_Regression_files/Image%20[1].png)
   - 分布：![](http://www.holehouse.org/mlclass/06_Logistic_Regression_files/Image%20[2].png)
4. 决策边界（decision boundary）：区分概率值（0.5）对应的theta值=0，所以函数=0所对应的线。
   - 线性区分的边界：![](http://www.holehouse.org/mlclass/06_Logistic_Regression_files/Image%20[5].png)
   - 非线性区分的边界：![](http://www.holehouse.org/mlclass/06_Logistic_Regression_files/Image%20[6].png)
5. 损失函数:
   - 问题：
   - ![](http://www.holehouse.org/mlclass/06_Logistic_Regression_files/Image%20[8].png)
   - 如果延续线性函数的损失函数，则可写成如下，但是当把逻辑函数代入时，这个损失函数是一个非凸优化（non-convex，有很多局部最优，难以找到全局最优）的函数。
   - ![](http://www.holehouse.org/mlclass/06_Logistic_Regression_files/Image%20[9].png)
   - 因此，需要使用一个凸函数作为逻辑函数的损失函数：
   - ![](http://www.holehouse.org/mlclass/06_Logistic_Regression_files/Image%20[12].png)