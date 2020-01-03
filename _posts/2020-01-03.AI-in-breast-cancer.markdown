---
layout: post
category: "genomics"
title:  "AI在乳腺癌检测中的应用"
tags: [genomics, machine learning]
---

### 目录

- TOC
{:toc}

---

### 背景

* 全球挑战：2018年造成100万人死亡（女性第二致死率癌症）
* 世卫组织建议项目：X线钼靶筛查
* 缺点：高的假阳性和假阴性；解读成本高；

---

### 模型：谷歌+DeepMind

[![20200103140024](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200103140024.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200103140024.png)

* 数据集的来源，这里只是验证集，不包含训练集
* 如何确定ground truth？通过预后跟踪，三个月的缓冲期
* 三种评估所建模型的效果

[![20200103140358](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200103140358.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200103140358.png)

* 跟临床医生的判断进行比较
* 左边是在UK测试集上的效果
* 右边是在USA测试集上的效果，模型训练时是否包含了UK数据集（实线：包含，虚线：不包含）

[![20200103140729](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200103140729.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200103140729.png)

对这个工作褒贬不一：

* 之前有发表过类似的工作，更大的验证，更好的模型效果（但是没有引起这么高的关注）
* 这个工作的代码没有公布。文章结尾只说了详细描述了如何复现，但是很多tool依赖于谷歌自己的框架。
* 过分夸大AI的效果，可以辅助医疗，但是远未达到完胜的程度。

---

### 参考

* [Nature发布AI检测乳腺癌最新成果，由谷歌、DeepMind联合开发，表现超过医生！](https://mp.weixin.qq.com/s/xE9-Y08k2fXmLEoRqtD2mA)
* [Nature发表Google新型AI系统！乳腺癌筛查完胜人类专家](https://mp.weixin.qq.com/s/RAjbQKTcgzPl48dYRrKK-w)
* [登上Nature却被打脸？LeCun对谷歌乳腺癌研究泼冷水：NYU早有更好结果](https://mp.weixin.qq.com/s/txtgkGNborrbi1KbtYigmA)
* [全球女性福音！DeepHealth深度学习模型检测乳腺癌完胜5名放射科医师](https://zhuanlan.zhihu.com/p/100551873)