---
layout: post
category: "machinelearning"
title:  "【4-4】特殊应用：人脸识别和神经风格转换"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

### 什么是人脸识别？

* 工作刷脸通过：人脸识别+活体检测
* 人脸验证：
	* face verification
	* 输入：一张图片，某个ID或者名字
	* 输出：输入图片是否是这个人
	* 做法：1对1问题，弄明白这个人是否和他声称的身份相符
* 人脸识别：
	* face recognition
	* 输入：一张图片
	* 输出：这个图片是否是某数据库中的哪个人
	* 做法：1对多问题
	* 正确率需要很高才能取得好的效果 [![20191009184259](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009184259.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009184259.png)

---

### One-Shot学习

* 人脸识别挑战：
	* 一次学习问题
	* 需要通过单单一张图片或者一个人脸样例就能去识别这个人
	* 例子：比如左边的是数据库的4个样本，对于新的人脸，如右侧的样本1、2，需要识别出1是数据库中的一个人，而2不是数据库中的某个人 [![20191009184605](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009184605.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009184605.png)
* 解法1：卷积神经网络【不推荐】
	* 比如数据库4个人，可以训练CNN，输出是4种
	* 效果不会好
	* 训练数据集太小，不能得到稳健的模型
	* 不方便：加入新加入成员，需要重新训练模型，因为此时CNN的输出是5种
* 解法2：**学习相似性函数**
	* 同样使用神经网络，但是学习目标发生变化
	* 学习相似性函数d：评估图片之间的差异大小的
	* 如果两张图片是同一个人，则相似性很高，差异很小，输出一个很小的值
	* 如果两张图片不是同一个人，则相似性很低，差异很大，输出一个很大的值 [![20191009185139](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009185139.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009185139.png)
	* 预测：输入一个新的图像，和数据库图像之间逐个比较，就知道这个人是哪个了

---

### Siamese网络

* 相似性函数d：输入两张人脸，告诉其相似度
* 做法：
	* 图片1通过网络编码为向量1
	* 图片2通过网络编码为向量2 
	* 两个向量之间的距离定义为两个图片的差异 [![20191009191827](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009191827.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009191827.png)
	* 同一个神经网络
	* 神经网络的参数定义了一个编码函数，参数不同，编码的结果不同。通过反向传播，获得最优的参数。[![20191009192348](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009192348.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009192348.png)

---

### Triplet损失

* 参数学习：定义三元组损失函数然后应用梯度下降
* 三元组损失：
	* 三张图片：Anchor图片（A），Positive图片（P），Negative图片（N）
	* 两个距离：A、P之间的距离，A、N之间的距离
	* 希望满足AP之间的距离小于AN之间的距离：$$\|f(A)-f(P)\|^2 <= \|f(A)-f(N)\|^2$$。比如下图中，AP是同一个人，AN不是同一个人，所以希望前者很接近，后者很不接近。[![20191009192824](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009192824.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009192824.png)
* 模型偷懒：
	* 需满足：$$\|f(A)-f(P)\|^2 - \|f(A)-f(N)\|^2 <= 0$$
	* 假如f函数对于任意的图片编码，总是为0，那么上面的式子就是：0-0<=0，是总是成立的
	* 为了防止网络输出无用的结果，可以🙆两者的差值不是小于0，而是一个更小一点的负数，即：$$\|f(A)-f(P)\|^2 - \|f(A)-f(N)\|^2 <= -a$$
	* 这里的-a叫做间隔（margin），类似于SVM里面的
* 例子：
	* 假如a=0.2，d(A,P)=0.5，如果d(A,N)=0.51，只比d(A,P)大了一点点，但是不满足两者之间的最小间隔，所以此时模型不够好。
* 损失函数：
	* 取0和上面等式值的最大值
	* max函数的作用：只要前半部分的值<=0，那么损失就是0，所以模型会想办法让前半部分更小 [![20191009193838](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009193838.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009193838.png)
* 三元组图片的选取：
	* 随机选取。在满足AP是同一人，AN不是同一人的条件下，随机选取图片组成三元组，此时是很容易满足$$\|f(A)-f(P)\|^2 - \|f(A)-f(N)\|^2 <= -a$$这个的。因为随机时，AN差别比AP差别很大的概率很大。此时网络不能学习到什么。
	* 构造难的三元组。所以应该选取比较难以区分的三元组，比如AP的距离和AN的聚集近似相等的。为了保证他们之间相差一个间隔，模型会尽力使得后者的距离变大，或者AP的距离变小。[![20191009194715](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009194715.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009194715.png)
	
---

### 人脸验证与二分类

* 做法：
	* 神经网络对于输入的两个图片进行特征编码
	* 将特征向量输入到逻辑回归单元，进行预测
	* 如果是相同的人，输出1；不同的人，输出0
	* 如此：把人脸识别问题转换为二分类问题 [![20191009211914](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009211914.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009211914.png)
* 如何使用逻辑回归？
	* 输入是两个图片，对应两个编码的特征向量
	* 这两个特征向量之间是有关系的
	* 可以**使用对应特征索性的差值**，输入到逻辑回归单元
	* 也可以使用其他的变换，比如x平方相似度输入到逻辑单元
	* 最后的输出是0（不同人）或者1（同一人） [![20191009212125](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009212125.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009212125.png)
* 训练集：
	* 监督学习，需创建只有成对图片的训练集
	* 标签1：这一对图片是同一人
	* 标签0：这一对图片是不同的人 [![20191009212430](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009212430.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009212430.png)
	
---

### 神经风格迁移

* C：内容图像
* S：风格图像
* G：生成的图像 [![20191009215935](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009215935.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009215935.png)
* 为什么可以做到？需要理解CNN由浅层到深层的特征表示，逐渐的都学习到了什么！

---

### CNN特征可视化

* 做法：
	* 对于某一层网络，选取一个神经单元，然后找到使得神经单元激活最大的一些（比如9个）小块 
	* 对于其他的一些神经元重复上面的操作
* 例子：这里选取的是layer1的神经元，对每个神经元，选取9个其最大激活的patch，然后画出来。这里相当于是选取了9个神经元，每一个含有9个patch。
	* 可以看到，不同的神经元激活得到不同的特征，比如有的是针对右斜45‘线条的，有的是左斜45’线条的。有的是一些边缘，有的是一些颜色阴影。每一个不同的图片块都最大化激活了。 [![20191009214526](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009214526.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009214526.png)
* 其他层：
	* 同样的，可以对其他层，比如更深的层进行同样的操作，查看所在层提取的特征
	* **更深的层一般可以检测到更加复杂的特征**
	* 比如这里的第二层：编号1检测垂直图案，编号2检测有圆形的，编号3检测很细的垂线
	* 第三层：编号1对图片左下角的圆形很敏感，编号2检测人类，编号3检测特定蜂窝状图案
	* 第四层：可以检测到更复杂的，编号1检测狗狗，很类似的狗狗，编号3检测到鸟的脚
	* 第五层：复杂的事物，编号1检测不同的狗，编号2检测键盘或类似物，编号3检测文本，编号4检测话 [![20191009215311](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009215311.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009215311.png)

---

### 代价函数

* 给定C和S，生成G。那么需要有一个函数去评价G的好坏！
* 关于G的函数：
	* 表示：$$J(G)=\alphaJ_{content}(C,G)+\betaJ_{style}(S, G)$$
	* 内容代价：度量生成图片G与内容图片C的内容有多相似
	* 风格代价：度量生成图片G的风格与图片S的风格的相似度
	* 两者之间的权重由两个超参数与来确定
* 算法：
	* 随机初始化生成G
	* **使用梯度下降的方法，使得代价函数J(G)最小化，更新G，也就是更新图像G中的像素值**
	* 经过不断的迭代之后，会生成一个越来越具有特定风格的图像G [![20191009221534](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009221534.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191009221534.png)

---

### 内容代价函数

### 参考

* [第四周 特殊应用：人脸识别和神经风格转换](http://www.ai-start.com/dl2017/html/lesson4-week4.html)

---




