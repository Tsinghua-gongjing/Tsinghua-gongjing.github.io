---
layout: post
category: "machinelearning"
title:  "【5-3】序列模型和注意力机制"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

### 基础模型

* seq2seq模型：
	* sequence to sequence
	* 机器翻译
	* 语音识别
* 例子：机器翻译【seq2seq】
	* 输入：法语句子
	* 输出：英语句子
	* 模型：
	* 编码网络：RNN结构，可向其中输入法语句子，接收完法语之后，会输出一个向量表示这个输入的法语句子
	* 解码网络：编码的输出作为输入，训练输出一个个的词，每个预测的词作为下一个的输入进行预测 [![20191012204131](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191012204131.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191012204131.png)
* 例子：图片描述【image to sequence】
	* 输入：一张图片
	* 输出：图像描述
	* 模型：
	* 编码网络；预训练的图像表示网络，用于提取图片的特征
	* 预测网络：基于图片特征向量，生成描述，可用RNN模型，最后会输出一个序列

### 选择最可能的句子

### 集束搜搜

### 改进集束搜索

### 集束搜索的误差分析

---

### 参考

* [第三周 序列模型和注意力机制](http://www.ai-start.com/dl2017/html/lesson5-week3.html)

---




