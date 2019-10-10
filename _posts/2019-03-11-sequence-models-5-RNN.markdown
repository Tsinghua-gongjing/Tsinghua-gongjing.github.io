---
layout: post
category: "machinelearning"
title:  "【5-1】循环序列模型"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

### 为什么选择序列模型？

* 在语音识别、自然语言处理等领域引起变革
* 例子：
	* 语音识别：输入音频片段，输出文字
	* 音乐生成：输入空集、数字等，输出序列
	* 情感分类：输入语句，输入对应的评价等级
	* DNA序列：输入DNA序列，输出匹配的蛋白质
	* 机器翻译：输入一种语言，输出另一种语言
	* 视频行为：输入视频祯，输出识别出的行为
	* 名字识别：输入语句，输出不同实体的名称 [![20191010143703](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010143703.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010143703.png)
* 注意：
	* 问题归纳：都是使用标签数据（X，Y）作为训练集的监督学习
	* 输入和输出不一定都是序列
	* 输入和输出的长度不一定相等

---

### 数学符号

* 都是序列模型，以输入是序列为例看怎么表示数据
* 例子：
	* 命名实体识别问题
	* 查找不同类型的文本中的人名、公司名、时间、地点、国家名和货币名等
	* 输入数据：$$x^{<t>}：x^{<1>},x^{<2>}...$$如此表示顺序索引的单词
	* t表示索引序列的位置，不管输入是不是时序序列，均可这样表示
	* 输入序列长度：$$T_x$$
	* 输出数据：$$y^{<t>}：y^{<1>},y^{<2>}...$$
	* 输出序列长度：$$T_y$$
	* 加入样本信息：$$x^{(i)<t>}$$，训练样本i的序列中的第t个元素
	* 序列长度：$$T_y^{(i)}$$，第i个样本的输出序列的长度 [![20191010144534](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010144534.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010144534.png)
* 单词表示：
	* 词典表示法
	* 每个单词使用所选词典的one-hot表示，此词所在index为1，其他index为0
	* 所以如果选用的词典是10000，那么对于一个序列长度为9的句子，其表示就是：9个长度为10000的向量
	* 遇到不在词典中的单词，可创建新的标记叫做Unknown Word的伪造单词，<UNK>作为标记 [![20191010144812](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010144812.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010144812.png)

---

### RNN模型

* 为什么不使用标准神经网络模型？
	* 对于序列问题，为什么不使用标准的神经网络模型，一层层的计算？
	* 问题1：不同的样本其长度不一定是相同的。不是所有的例子都有着相同的输入长度或者输出长度。标准网络的表达方式不太好。
	* 问题2：从不同位置学到的特征不能共享。比如从某个例子中学到位置1出现的词是人名，那么这个词出现在其他的位置，应该也能学到其是人名。
	* 问题3：参数巨大。比如词向量进行one-hot编码，如果词典大小是10000，那么每个词的表示就是这么大，一个10个单词的句子就是10x10000大小。全连接下的参数会非常巨大。[![20191010161907](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010161907.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010161907.png)
* RNN：
	* 每一个时间步，RNN网络传递一个激活值到下一个时间步用于计算
	* 零时刻：构造一个激活值，通常是零向量
	* 每个时间步的参数是共享的 [![20191010164827](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010164827.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010164827.png)
	* 单向RNN：只用了序列中之前的信息做出预测
	* 双向RNN：可以用到前后的序列信息做出预测
	* 例子：上面的语句中，如果只根据前面的单词信息，不知道 Teddy 究竟是什么，但是使用前后的信息后，就知道这是不是个人名了。
* RNN前向传播计算：
	* 根据每个时间点的输入，前一个时间点的激活值，来计算当前时间点的激活值和预测值 [![20191010165327](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010165327.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010165327.png)	
	* RNN中常用的激活函数是tanh
	* 计算的简化表示：对于前一时间点的激活值要乘以一个参数，当前时间点的输入也要乘以一下参数，所以可以把这部分进行合并表示（就是下图中的编号1可表示为编号2，编号345是拆开了说为什么是相等的）[![20191010165641](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010165641.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010165641.png)
* 前向传播示意图：
	* 这里是对每个cell展开了具体的计算流程示意图
	* **先计算激活值，再根据激活值计算预测值**。时间t时刻的预测值是依赖于激活值的计算的。[![20191010170203](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010170203.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010170203.png)

---

### 通过时间的反向传播

* 因为这里的网络是按照时间进行传递的，所以反向传播相当于时间回溯了
* 下图中蓝色的流就代表前向传播，红色的流代表反向传播 [![20191010170736](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010170736.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010170736.png)
* 例子：计算a1，y1
	* 输入：x1
	* 前一刻的激活值：a0
	* **计算a1还需要：Wa和ba**，因为a1=g(Wa * [a0, x1] + ba)，这里的Wa和ba在之后的每一个时间步也是需要用到的，所以是需要学习的参数
	* **计算y1还需要：Wy和by**，因为y1=g(Wy * a1 + by)，这里的Wy和by在之后的每一个时间步也是需要用到的，所以是需要学习的参数
* 损失函数：
	* 逻辑回归损失函数（交叉熵损失函数）
	* 整个序列的损失：每一个时间步上的损失的和
* 反向传播示意图：
	* 总损失下使用梯度更新参数 [![20191010171618](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010171618.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010171618.png)

---

### 不同类型的RNN

* [![20191010171701](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010171701.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010171701.png)
* 一对一：标准的小型神经网络，输入x输出y
* 一对多：一个输入，多个输出
	* 音乐生成。输入只有一个，比如音乐类型等，甚至是空的输入，但是需要输出一连串的音符
* 多对一：很多的输入，最后一个时间上输出
	* 情感分析。输入的是一个序列（句子），最后给出一个评价等
* 多对多：输入是一个序列，输出也是一个序列
	* 输出和输入长度相等。比如实体名称标注。
	* 输出和输入长度不相等。比如机器翻译，从一个语言翻译到另一个语言，所用的词语长度不一定是一样的。

---

### 语言模型和序列生成

* 语言模型：
	* 语音识别系统：计算每句话出现的可能性
	* **告诉某个特定的句子出现的概率** [![20191010214852](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010214852.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010214852.png)
* 如何构建语言模型？
	* 训练集：很大的英文文本语料库（corpus）
	* 语料库：自然语言处理的专有名词，指很长的或者说数量众多的英文句子组成的文本
	* 句子标记化：根据词字典（比如10000个词），把句子中的每个单词转化成one-hot向量
		* 定义句子结尾：增加额外的标记EOS
		* 标点符号：自行决定是否看成标记
		* 词不在词典中：用UNK进行替换，表示未知词
		* 完成：输入的句子都映射到字典中的各个词上了 [![20191010215349](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010215349.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010215349.png)
	* 构建RNN模型得到这些序列的概率
		* 时间t=0：输入x1和a0计算激活项a1，通常a0取0向量，x1也设为全为0的集合。此时是通过softmax预测第一个词是什么（就是y1），具体就是预测任意词为第一个词的概率，比如a，the，dog等各个词的概率。
		* 时间t=1：计算第二个词（y2）是什么。有激活项a1，x1是什么？此时的x1是上一个时刻的输出y1，即x2=y1，那么久可以计算激活项a2，然后计算y2。此时预测y2是各个词的概率，只是此时的概率是个条件概率，就是说前面已经是某个词的情况下，此时预测的词是什么。
		* 依次类推，把上一个时刻预测的输出作为下一个时刻的输入，再预测当前时刻的输出。每一步都会考虑前面得到的单词。
		* 直到到达序列结尾 [![20191010220338](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010220338.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010220338.png)
* 代价函数：
	* softmax损失：预测值和真实值之间的差异
	* 总体损失：单个预测的损失函数相加起来
	
---

### 对新序列采样

* 如何知道模型学到了什么？新序列采样的方式进行评估
* 采样：
	* 第一个词。模型已经知道了第一位置出现的各个词的概率，那么可以根据这个向量的概率分布进行采样。
	* 把采样得到的第一个词作为下一个时间的输入，然后预测得到y2
	* 再把y2作为下一个时间的输入，以此类推
	* 最后就可以得到一个完整的句子
	* 基于词汇的RNN模型（字典中的词是英语单词） [![20191010223152](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010223152.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010223152.png)
* 基于字符的RNN模型
	* 字典：字母、大小写的、数字
	* 训练数据：单独的字符而不是词汇
	* 优点：不必担心会出现未知的标识
	* 缺点：会得到太多太长的序列，计算成本昂贵 [![20191010224154](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010224154.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010224154.png)
* 绝大部分使用的是基于词汇的语言模型
	* 基于字符的语言模型
	* 左边：新闻文章训练出来的
	* 右边：莎士比亚的文章训练出来的 [![20191010224350](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010224350.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191010224350.png)

---

### 参考

* [第一周 循环序列模型](http://www.ai-start.com/dl2017/html/lesson5-week1.html)

---




