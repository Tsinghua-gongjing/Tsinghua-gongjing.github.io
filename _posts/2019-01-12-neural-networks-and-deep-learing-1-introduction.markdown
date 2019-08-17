---
layout: post
category: "machinelearning"
title:  "【1-1】深度学习引言"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

### 简介

* AI是最新的电力
* 深度学习：AI中发展最为迅速的分支
* 课程1：神经网络与深度学习
	* 神经网络基础
	* 建立神经网络
	* 在数据上训练
	* 用一个深度神经网络进行辨认猫
* 课程2：
	* 深度学习实践
	* 严密构建神经网络
	* 使得表现良好：超参数调整、正则化、诊断偏差和方差
	* 高级优化算法：momentum、Adam
* 课程3：
	* 结构化机器学习工程
	* 构建机器学习系统能改变深度学习的错误
	* 端对端深度学习
* 课程4：
	* CNN：应用于图像领域
* 课程5：
	* 序列模型
	* 应用于自然语言处理
	* RNN、LSTM

---

### 什么是神经网络

* 预测房价：
	* 基于房屋面积
	* 线性回归
	* 房价不可能为负
	* 所以让直线变得“弯曲”了一点
	* 这条线对应于神经网络中的激活函数ReLU
	* ReLU：Rectified Linear Unit，可以理解为max(0,x)。具有修正作用。【**预测房价引入修正功能的激活函数**】![](http://www.ai-start.com/dl2017/images/3fe6da26014467243e3d499569be3675.png)
* 房价预测的更复杂的神经网络：
	* 同样是房价预测，其实很多其他的因素对于房价也具有影响
	* 考虑：size,bedroom,zip code,wealth ![](http://www.ai-start.com/dl2017/images/7a0e0d40f4ba80a0466f0bd7aa9f8537.png)
	* 隐藏单元：比如第一个节点代表家庭人口，这个只与size和bedroom特征有关，通过权重来实现
	
	* 神经网络：擅长计算从x到y的函数映射

--

### 神经网络的监督学习

* 神经网络创造的经济价值，本质上离不开监督学习
* 很多监督学习的例子 ![](http://www.ai-start.com/dl2017/images/ec9f15da25c4072eeedc9ba7fa363f80.png)
* 在线广告：
	* 深度学习最获利
	* 网站中输入广告的相关信息
	* 输入用户的信息
	* 考虑是否展示广告
* 计算机视觉：
	* 给照片打标签
* 语音识别：
	* 机器翻译：中英文
	* 语音转换为文字
* 自动驾驶


* 标准的神经网络：房地产、在线广告
* CNN：图像
* RNN：序列数据，如音频、语言


* **结构化 vs 非结构化**：
	* 结构化数据：诶个特征有一个很好的定义，基本数据库等
	* 非结构化数据：音频、图像、文本，特征可能是像素值或者单个单词 ![](http://www.ai-start.com/dl2017/images/86a39d40cb13842cd6c06463cd9b4a83.png)
* 计算机：善于理解结构化数据
* 人：善于解读非结构化数据

---

### 为什么深度学习会兴起

* **数据规模增大**
	* 数字化的数据
	* 超过传统机器学习算法的能力
	* 好的性能：【1】训练足够大的神经网络，【2】需要很多的数据 ![](http://www.ai-start.com/dl2017/images/2b14edfcb21235115fca05879f8d9de2.png)
* **算法的创新**：
	* 巨大突破：从sigmoid函数转换到ReLU函数。sigmoid的梯度会接近于0，从而参数更新非常缓慢，速率变得很慢。
	* 更多的影响：计算的优化。提出想法到实现想法，检查结构时间更短。
* **计算能力的提升**
* ![](http://www.ai-start.com/dl2017/images/e26d18a882cfc48837118572dca51c56.png)

---

### 参考

* [第一周：深度学习引言](http://www.ai-start.com/dl2017/html/lesson1-week1.html)

---




