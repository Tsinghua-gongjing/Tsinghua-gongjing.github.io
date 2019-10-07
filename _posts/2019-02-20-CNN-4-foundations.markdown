---
layout: post
category: "machinelearning"
title:  "【4-1】卷积神经网络"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

### 计算机视觉

* 深度学习的兴奋点：
	* （1）CV的高速发展标志着新型应用产生的可能
	* （2）新的神经网络结构与算法，启发创造CV和其他领域的交叉成果
* 例子：
	* 图片分类（图片识别）：识别出一只猫
	* 目标检测：物体的具体位置，比如无人车中车辆的位置以避开
	* 图片风格迁移：把一张图片转换为另一种风格的
* 挑战：
	* 数据量大：3通道64x64大小（64x64x3=12288）
	* 特征向量维度高，权值矩阵或者参数巨大
	* 参数巨大：难以获得足够的数据防止过拟合
	* 巨大的内存需求

---

### 边缘检测示例

* 边缘检测例子：
	* 可能先检测竖直的边缘
	* 再检测水平的边缘 [![20191006163223](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191006163223.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191006163223.png)
* 如何检测？
	* 使用过滤器(核，filter)
	* 通过一个过滤器去和图片进行卷积，得到其卷积结果6x6的矩阵和3x3的过滤器卷积运算后得到4x4的新矩阵 
	* 左边：图片，中间：过滤器，右边：另一张图片。这里的过滤器其实就是垂直边缘检测器。[![20191006163850](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191006163850.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191006163850.png)
* 为啥是**垂直边缘检测器**？
	* 换个图片例子：原始图片大小还是6x6的，但是里面的像素值变成现在的样子
	* 原始图片：左侧是10，右侧全为0，对应的图像就是左侧近乎于白色（像素值大，比较亮），右侧近乎于灰色（像素值小，比较暗）。这里左右两侧的中间可以看到一个明显的垂直边缘，达到从白到黑的过度。
	* 过滤器：此时左侧全为1（最亮，白色），中间全为0（一般，灰色），右侧全为-1（最小，黑色）。
	* 卷积后：得到的图像左右两边全为0，中间为30，对应图像是左右两边近乎灰色（偏暗），中间是白色（偏亮）。
	* 中间的两列其实对应的就是整个图像的一条亮线，只是由于这里图片太小（4x4），所以看不太出来。如果是1000x1000的，那么可以明显看到卷积之后是可以看到正中间有一条垂直边缘亮线的。
	* 因此说：此时用到的这个过滤器可以检测图片中的额垂直边缘。 [![20191006165013](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191006165013.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191006165013.png)
* 过度的变化可被检测到：
	* 上面原始图片是从亮（10）到暗（0）的过度，如果换成从暗（0）到亮（1）的过度呢？检测得到的图片是怎么样的？
	* 可以看到：得到的图片中间被翻转了，不再是原来的由亮（白色）向暗过度，而是由暗（黑色）向亮过度。如果不考虑两者区别，可取绝对值。 [![20191006165804](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191006165804.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191006165804.png)
* 垂直检测与水平检测：
	* 垂直：1 =》 0 =》 -1，左边亮，右边暗
	* 水平：1 =》 0 =》 -1，上面亮，下面暗 [![20191006165921](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191006165921.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191006165921.png)
* 衍生：
	* 不同的3x3的滤波器：[![20191006214116](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191006214116.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191006214116.png)
	* Sobel过滤器：增加中间一行元素的权重，使得结果具有更高的鲁棒性
	* Scharr过滤器：属于一种垂直边缘检测，如果将其翻转90度，就能得到对应水平边缘检测
* **为什么可以学习到图像的特征？**
	* 上面列举的可以检测到垂直或者水平的边缘
	* 还有的其他过滤器可以学习45°、70°或者73°，甚至是任何角度的边缘
	* 矩阵所有数字设置为参数，通过数据反馈，让神经网络自动学习，就可以学习到一些低级的特征，比如各种不同角度的边缘特征等。
	* 过滤器数字参数化的思想，是CV中最为有效的思想之一

---

### Padding

* 直接卷积的缺点：
	* **输出缩小**。每次卷积操作之后，图像就会缩小，比如从开始的6x6变为4x4，最后变为1x1等。
	* **丢掉了图像边缘位置的许多信息**。在角落或者边缘区域的像素点在输出中采用较少。
* 解决上述问题：
	* 卷积前填充
	* p是填充数量，习惯用0进行填充。填充后的输出：(n+2p-f+1)x(n+2p-f+1)，此时输出的图像和原来可能更接近了。
	* 边缘发挥小作用的缺点也被弱化。
* 填充像素p：
	* Valid卷积：不填充。输出：(n-f+1)x(n-f+1)
	* Same卷积：填充后输出大小和输入大小一样。输出：(n+2p-f+1)x(n+2p-f+1)，此时前后大小相等有：n+2p-f+1=n => p=(f-1)/2。所以当f为奇数时，选择合适的填充p大小，可以使得卷积前后大小相等。 ![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191006235909.png)
* 为什么过滤器通常是奇数？
	* 习惯：f通常为奇数。
	* 如果f是偶数，只能使用一些不对称的填充
	* 奇数过滤器有一个中心点，便于指出过滤器的位置

---

### 卷积步长

* 基本操作：步幅（stride）
* 例子：一个7x7的图像，使用一个3x3的过滤器，步幅f=2，padding=0 ![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007000228.png)
* 公式：
	* 输出：$$(\frac{n+2p-f}{s}+1) \times (\frac{n+2p-f}{s}+1)$$
* 如果商不为整数怎么办？
	* 向下取整：对z进行地板除
	* 惯例：只有上面的每一个篮框完全包括在图像或填充完的图像内部时，才对它进行运算。如有有任意的一个篮框移动到了外面，那么就不要进行相乘操作。
	* 输出维度总结：![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007000749.png)

---

### 三维卷积

* 过滤器是三维的，对应RGB三个通道
* 例子：
	* 输入：6x6x3的图像
	* 过滤器：3x3x3
	* 输出：4x4x1.注意不是4x4x3，对于每一个过滤器立方体，总共有27个数字，这27个数字对应相乘然后求和得到一个数字，称为新图像的一个元素。 ![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007004352.png)
	* 应用：比如如果绿色和蓝色通道全为0，红色通道设置一个垂直检测器，那么整个三维的过滤器就是一个只对红色通道有用的垂直边界检测器。![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007004755.png)
* 多过滤器：
	* 一个三维立体过滤器实现一个特征检测
	* 同时使用多个过滤器检测到不同的特征 ![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007004935.png)
* 输出维度：
	* 输入：n x n x nc(通道数)
	* 卷积：f x f x nc，惯例nc是相等的
	* padding：0
	* stride：1
	* 输出：(n-f+1) x (n-f+1) x nc'，nc'是下一层的通道数，也就是使用的过滤器的个数。

---

### 单层卷积网络

* 如何构建卷积？
* 根据输入图像，卷积得到卷积值，然后经过激活函数。
* 示例：a0到a1的演变：
	* 先执行线性函数
	* 所有元素相乘做卷积：运用线性函数再加上偏差
	* 应用激活函数ReLU ![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007005816.png)
* 参数个数？
	* 输入：1000x1000x3
	* 过滤器：10个3x3x3的
	* 参数：每一个过滤器是28个参数（3x3x3+1偏差），10个过滤器就是280个参数。【参数个数与输入图像的大小是无关的】
	* 不管输入图片大小，参数都很少，这是卷积神经网络的避免过拟合的特性。
* 卷积层各种标记：
	* 输入
	* 过滤器大小
	* padding
	* stride
	* 过滤器数目
	* 输出：通道数量就是此层中过滤器的数量  ![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007010453.png)
* 如何确定权重参数？
	* 参数就是每个过滤器的数值的权重 
	* 数据拟合

---

### 简单卷积网络示例

* 通常一个卷积神经网络包含3层：
	* 卷积层：CONV
	* 池化层：POOL
	* 全连接层：FC
* 一个简单的卷积网络示例：
	* 这里是展示了几个卷积层
	* 通过不同的卷积层之后图片的大小、每个卷积的设置等 [![20191007125635](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007125635.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007125635.png)
	* 【输入-卷积层1】：
		* 输入：39x39x3
		* 卷积大小：f=3
		* 步幅：s=1
		* 填充：p=0.（不填充，属于valid卷积）
		* 卷积数目：10
		* 输出：37x37x10，37=(n+2p-f)/s + 1 = (39+0-3)/1 + 1=37, 10是使用10个过滤器，所以输出会有10个通道（三维堆叠，每一层是一个过滤器）
	* 【卷积层1-卷积层2】：
		* 输入：37x37x10
		* 卷积大小：f=5
		* 步幅：s=2
		* 填充：p=0.（不填充，属于valid卷积）
		* 卷积数目：20
		* 输出：17x17x20，37=(n+2p-f)/s + 1 = (37+0-5)/2 + 1=17, 20是使用20个过滤器，所以输出会有20个通道（三维堆叠，每一层是一个过滤器）
	* 【卷积层2-卷积层3】：
		* 输入：17x17x20
		* 卷积大小：f=5
		* 步幅：s=2
		* 填充：p=0.（不填充，属于valid卷积）
		* 卷积数目：40
		* 输出：7x7x40，37=(n+2p-f)/s + 1 = (17+0-5)/2 + 1=7, 40是使用40个过滤器，所以输出会有40个通道（三维堆叠，每一层是一个过滤器）
	* 经过3个卷积之后，为图片提取7x7x40=1960个特征。可对提取的特征进行处理，比如展开向量，然后过逻辑回归或者softmax回归等。

---

### 池化层

* 池化层：
	* 缩减模型大小
	* 提高计算速度
	* 提高所提取特征的鲁棒性
* 输出大小：
	* 公式和前面的卷积一样
	* 池化也是需要顶一个过滤器的大小、步幅、填充的
	* 输出大小即：(n+2p-f)/s + 1 
	* 计算神经网络某层的静态属性，**没有参数需要学习，只有一组超参数（过滤器大小f和步幅s）** [![20191007130923](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007130923.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007130923.png)
* **最大池化**：
	* 在任何一个象限内提取到某个特征，保留其最大值
	* 很少用到超参数padding，p=0最常用
* 三维池化：
	* 输入是三维的，那么输出也是三维的
	* 对每个通道分别执行池化操作 [![20191007131241](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007131241.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007131241.png)
* **平均池化**：
	* 选取每个过滤器的平均值
	* 不太常用 [![20191007131601](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007131601.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007131601.png)

---

### CNN示例

* 常见模式：
	* 一个或者多个卷积层跟随一个池化层
	* 然后一个或者多个卷积层再跟随一个池化层
	* 然后是几个全连接层
	* 最后是一个softmax
* 一个跟LeNet5很像的CNN：[![20191007132118](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007132118.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007132118.png)
	* Layer1：一个卷积+一个池化。也有的把他们作为单独的层。
	* 计算神经网络多少层：通常只统计具有权重和参数的层，池化层没有权重和参数（只有超参数）
	* 尽量不要自己设置超参数，而是查看文献中别人采用了哪些超参数
* 激活值形状、大小、参数数量：
	* 比如上面的网络，可以列出下表 [![20191007132442](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007132442.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007132442.png)
	* 随着神经网络加深，激活值尺寸会逐渐变小
	* 池化层和最大池化层没有参数
	* 卷积层的参数相对较少

---

### 为什么使用卷积？

* 卷积层 vs 全连接层：
	* 参数共享：适用于图片的某个区域的检测器，也是用于图片的其他区域进行特征检测
	* 稀疏连接：卷积后的某个元素值，只与原来图像中的某些个数值有关（fxf个），而与其他的像素值无关 [![20191007132945](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007132945.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191007132945.png)
* 优点：
	* 减少参数
	* 使用更小的训练集训练
	* 预防过拟合
	* 善于捕捉平移不变：移动有限个像素，鉴定的猫依旧清晰可见

---

### 参考

* [第一周 卷积神经网络](http://www.ai-start.com/dl2017/html/lesson4-week1.html)

---




