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
	* 预测网络：基于图片特征向量，生成描述，可用RNN模型，最后会输出一个序列 [![20191013145628](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013145628.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013145628.png)

---

### 选择最可能的句子

* 语言模型：
	* **估计句子的可能性**
	* 比如给定中间的词，预测上下文的词
	* 比如给定上下文的词，预测中间的词
* 机器翻译：
	* 条件语言模型
	* 输入法语句子
	* **输出英语句子，估计一个英文翻译的概率。这个英语句子是相对于输入的法语句子的可能性，所以是一个条件语言模型**。[![20191013150236](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013150236.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013150236.png)
* 翻译概率最大化：
	* 输入：一个法语句子
	* 输出：一个翻译的英文句子
	* 目的：不是想随机的输出，想输出一个很好的翻译。你会有很多的翻译可能性，那么就是要**找到最合适的翻译y，使得这个条件概率（在给定法语句子时英语翻译的概率）最大化。**
	* 做法：集束搜索（beam search）[![20191013150732](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013150732.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013150732.png)
* 为什么不用贪心算法？
	* 如果这里用贪心算法，做法就是这样的：生成第一个词的分布，选择最可能的一个；挑选了第一个后，继续挑选第2个最有可能的，如此继续。每一步都是挑选最优的一个。
	* 翻译本质：挑选一个整体最优的，而这个最优的可能不是第一步最最优的，或者不是第二步是最优的。
	* 例子：看下面的两个翻译
	* 贪心：选择了”Jane is“之后，贪心会选择”going“，因为这个词是更常见的，概率是更大的。
	* 但是显然第1个翻译比第2个是更好的，所以根据贪心的算法，最后可能选择了一个欠佳的翻译。[![20191013151333](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013151333.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013151333.png)
	* 另外，每一步词的可选数量就是整个词典的大小，那么所有的都要进行计算的话，是不现实的。
* 近似搜索算法：尽力挑选出句子y使得条件概率最大，但不能保证找到的y值一定可以使得概率最大化。

---

### 集束搜索

* 核心：每一步都是选出B个最可能的单词，而不是一个
* 参数B：集束宽，beam width，考虑多少个选择
* 例子：
	* 【挑选第1个单词】
	* 在给定输入的情况下，获得了输入的表示向量。词向量输入到解码网络中，预测第1个词是什么。解码网络有一个softmax层，会得到10000个单词的概率值，那么取前B个（比如这里是3个）单词存下来便于后面使用。[![20191013152534](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013152534.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013152534.png)
	* 【挑选第2个单词】
	* 比如第1步选出来最有可能的3个词：in、jane、september
	* 接下来选择第2个词，怎么选？对于第一步选的3个词，分别作为y2的输入，预测此时对应的y2，选取其对应的概率值最大的3个。这样就得到了，输入法语句子+输出一个词特定的条件下，第二个词应该选择什么。[![20191013165157](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013165157.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013165157.png)
	* 【挑选第3个单词】
	* 类似的，对于上面的情况，分别进行讨论，看在对应的第1、第2个词的情况下，保留第3个最大的情况 [![20191013165313](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013165313.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013165313.png)
	* 不断进行搜索，一次增加一个单词，直到句尾终止。
* 集束搜索 vs 贪心算法：
	* 束宽的值可以选取不同的
	* 当束宽=1时，每次就只考虑一个可能结果 =》贪婪搜索算法

---

### 改进集束搜索

* 乘积概率 =》对数概率和：
	* 最大化的概率：乘积概率
	* 概率值小于1，且通常远小于1。很多数相乘，会造成数值下溢（数值太小，电脑的浮点表示不能准确存储）。
	* 乘积概率转为对数概率和：便于存储和计算
* 长度归一化：
	* 原来：长句子 =》概率值低，短句子 =》概率值高。
	* 缺点：可能不自然的倾向于简短的翻译结果，更偏向于短的输出，因为短句子的概率是由更少数目的小于1的数字乘积得到，所以不会那么小。
	* 归一化：除以翻译结果的单词数量，就是取每个单词的概率对数值的平均，可明显的减少对输出长的结果的惩罚。[![20191013170656](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013170656.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013170656.png)
	* **归一化的对数似然目标函数**
	* 探索性：在长度Ty上加一个指数a，a可以取0.7，1.0之类的。如果a=1.0，就是完全长度归一化，如果a=0，那么就是没有归一化。设置适当的值，可以获得不同的效果。
* 如何选取束宽值B？
	* B越大，考虑的选择越多，找到的句子可能越好，但是计算代价越大
	* B越小，考虑的选择越少，找到的句子可能不好，但是计算代价越小
	* 例子中使用的是3，实践中是偏小的
	* 在产品中可设置到10，100可能是偏大的
	* 束宽1变为3，模型的效果可能有所提升；束宽1000变为3000，模型的效果可能没有哦太多提升。

---

### 集束搜索的误差分析

* 误差分析：通过分析错误样本，定位模型的性能瓶颈，指导下一步应该从哪里进行改进
* 翻译模型：
	* 组成：seq2seq模型（RNN的编码和解码器） + 集束搜索模型（寻找可能最优的翻译）
	* 要么是RNN出错，要么是集束搜索出错，如何定位？

	* 输入：法语句子
	* 真实翻译：人的翻译$$y^*$$
	* 算法翻译：算法的输出$$\overline{y}$$
	
	* 做法：使用RNN模型计算两个概率，一个是在给定x下，人的翻译的条件概率：$$P(y^*\|x)$$；一个是在给定x下，算法的翻译的条件概率：$$P(\overline{y}\|x)$$
	* 判断：如果$$P(y^*\|x)$$大于$$P(\overline{y}\|x)$$，真实的人的翻译比算法选择的更好，但是最后搜索算法选择了现在的$$\overline{y}$$，说明是集束搜索算法出现了问题。
	* 判断：如果$$P(y^*\|x)$$小于$$P(\overline{y}\|x)$$，其实的情况是真实的人的翻译比算法选择的更好，但是算法确计算出前者概率更小，算法也确实是输出了基于RNN的更好的$$P(\overline{y}\|x)$$，所以算法没有问题，是RNN模型本身不够好。[![20191013172455](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013172455.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013172455.png)
* 误差分析：
	* 遍历数据集或者挑选一些例子
	* 对于每个例子，用RNN模型计算上面提到的概率值
	* 根据上面的判断原则，确定每个样本中是哪一个部分出现了问题：RNN还是beam搜索？
	* 最后统计一下两者的比例，就知道接下来应该优化哪一部分了 [![20191013172950](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013172950.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191013172950.png)

---

### 参考

* [第三周 序列模型和注意力机制](http://www.ai-start.com/dl2017/html/lesson5-week3.html)

---




