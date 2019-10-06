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

---

### 卷积步长

---

### 参考

* [第一周 卷积神经网络](http://www.ai-start.com/dl2017/html/lesson4-week1.html)

---




