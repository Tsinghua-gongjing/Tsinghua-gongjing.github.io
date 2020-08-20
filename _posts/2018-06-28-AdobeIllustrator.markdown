---
layout: post
category: "other"
title:  "Adobe Illustrator usage"
tags: [AI, plot]
---

# Adobe Illustrator (AI)

- TOC
{:toc}

---

## Install:

Refer to [Adobe Illustrator CC 2017 MAC中文破解版](http://www.web3.xin/soft/53.html)

---

## AI 官方用户指南 (中文版)

##### 简介

* [工作区基础知识](https://helpx.adobe.com/cn/illustrator/using/workspace-basics.html) 

* [创建文档](https://helpx.adobe.com/cn/illustrator/using/create-documents.html)

* [工具](https://helpx.adobe.com/cn/illustrator/using/tools.html)

---

#### 工作区

* [自定义工作区](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/customizing-workspace.html)

* [“属性”面板](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/properties-panel.html)

* [画板](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/using-multiple-artboards.html)

---

#### 绘图

* [绘图了解有关 Illustrator 中绘图工具的信息。
绘图基础知识](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/drawing-basics.html)

* [使用钢笔、曲率或铅笔工具绘制](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/drawing-pen-curvature-or-pencil.html)

* [绘制简单线段和形状](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/drawing-simple-lines-shapes.html)

* [绘制像素级优化的图稿](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/pixel-perfect.html)

* [编辑路径](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/editing-paths.html)

* [调整路径段](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/adjust-path-segments.html)

---

## 常规操作

* **裁剪图片**：
  - 操作：先在图片上画一个形状（比如长方形、圆圈等），然后同时选择该形状和图片，再`command+7`快捷键，即完成剪切操作。

* **改变点的大小**：
  - 操作：可以改变对象的大小、旋转角度等，比如可以用来改变散点图中点的大小
  - 路径：`object -> transform -> transform each`
  - 快捷键：`shift+option+cmd+d`
  
* **更改某个部分的对象**:
  - 操作：对于某个图像的局部进行更改，比如更改某个panel的图的点的大小
  - 具体：以改变某个三店图为例，首先选取要更改的部分，然后成组。再选中这个组，此时整体图上的其他部分应该是不可选的。接着选取此组中的某个点，选择相同的对象，则只会选中此组中的点对象，从而可以进行颜色更改、大小控制等操作。

* **释放剪切模板**：
  - 操作：对于一些点对象，看起来是个点，其实很能还有正方形的外圈等对象，这些对象容易影响后面对于点对象的操作，可以先去除这些对象。一般先选择某个点，然后释放剪切模板，拖拉把除了点以外的对象选中，选择具有相同属性的其他对象，全部去除。
  - 路径：`select -> same -> fill & stroke`
  - 快捷键：`option+cmd+7`。
  
* **成组**：
  - 操作：把不同的对象group起来进行变换等
  - 路径：`object -> group`
  - 快捷键：`cmd+g`

* **解组**：
  - 操作：把成组的对象进行解组
  - 路径：`object -> ungroup`
  - 快捷键：`shift+cmd+g`

* **画透明的圆圈**：
  - 操作：画一个圆圈，或者其他形状，设置为透明的，方便相互重叠，比如画韦恩图
  - 方案1：直接用shaper tool手动画一个椭圆，会自动识别生成一个圆圈。设置填充为None，线条选定颜色即可。
  - 方案2：直接用shaper tool手动画一个椭圆，会自动识别生成一个圆圈。在工作区上方有个不透明度（Opacity），可根据需要阈值进行设置，这个与上面的区别是线条也会受到这个阈值的调控，所以当设置不透明度为0（即完全透明）时，线条也是看不见的，即使设置线条很粗也没用。
  - 方案3：直接用shaper tool手动画一个椭圆，会自动识别生成一个圆圈。在右侧工具栏使用透明（transparent）工具，这个的问题和上面的不透明度（Opacity是一样的。

* **吸管工具**：
  - 【填充 -> 填充】吸取一个对象1的填充色作为对象2的**填充色**：按下`V键`，切换到`选择工具`，单击待填充对象2，然后按下`I键`，切换到`吸管工具`，鼠标变为一支吸管，在对象1单击，就可以将对象1的填充颜色复制填充到对象2上。
  - 【填充 -> 轮廓】吸取一个对象1的填充色作为对象2的**轮廓色**：按下`V键`，切换到`选择工具`，单击待填充对象2，单击工具箱中的`轮廓`，将`轮廓色`放在上方。按下`I键`，切换到`吸管工具`，鼠标变为一支吸管，`按着Shift键`，在对象1单击吸取颜色，这样就可以将吸取的颜色放在对象2的轮廓上了。
  - 【文字 -> 文字】：类似于上面的【填充 -> 填充】操作，只不过对象全部换为文字，可以吸取文字的大小、颜色、字体等熟悉。
  - 【吸取界面之外的颜色】：按下`V键`，切换到`选择工具`，单击待填充对象，然后按下`I键`，切换到`吸管工具`，鼠标变为一支吸管，**按住鼠标左键不放，将光标移到AI界面之外的任意目标颜色处，释放鼠标**，即可将吸取的颜色填充到需要改变颜色的图像上面。

 * **设置同一行换行文字间距**:
   - 选中要设置的文字，然后调成文本属性（`command+t`），设置文字：`VA加下划线`的，即可调整间距。可参考[这里](https://sunny0731.pixnet.net/blog/post/42051571-ai%E6%AA%94-%E5%AD%97%E5%85%83%E9%96%93%E8%B7%9D%E3%80%81%E6%AE%B5%E8%90%BD%E8%A1%8C%E8%B7%9D%E8%AA%BF%E6%95%B4) [![20190806115213](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190806115213.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190806115213.png)

* **文本框自动换行**：
  - 如果是正常的文字，是不能自动换行的。需要`画`一个文本框，具体就是：选中文本按钮，在画布中拖出一个一定长宽的文本框，那么这个框里面的文字是会自动换行的。

* **设置不同行文字间距（段落间距）**：
   - 选中要设置的文字段落，然后选择`段落(Paragraph)`属性，在这里可以设置文字的对齐方式，一般选择中间的两段对齐，多余行左对齐。在下方可以设置段落间距（类似于word里面的行间距）。 [![20190806115847](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190806115847.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190806115847.png)

* **整个画板变为灰色网格？**
	* 因为其默认的快捷键是`shift+cmd+D`
	* 有时候想想设置元素大小，使用快捷键`shift+option+cmd+d`，但是不小心点错了
	* 具体效果可以参考这里：[ai画板与背景整个变成灰色格子了？](https://zhidao.baidu.com/question/1796240806630924787)

* **更改画布大小**
	* 在右侧artboard里面选择画布
	* artboard界面中，右上角列表按钮下拉，选择`artboard options`
	* 设置宽和高
	* 也可以在顶部断则`Document Setup`，有`edit board`选项，也可设置
	* 具体可以参考这里：[ai怎么调整画布尺寸? ai设置画布大小的两种方法](https://www.jb51.net/Illustrator/454077.html)

* **统一多个长方形的大小**
	* 选择要选取的多个长方形
	* 先把这些选择的对象转换为形状：`对象` => `形状` => `转换为形状`
	* 然后设置形状的长宽，即可统一所有的大小

* **调整一段文字的行间距**
	* 选择要调整的文字
	* MAC下面：`option+→`:扩展字间距，`option+←`:缩小字间距，`option+↑`:缩小行距，`option+↓`:扩展行距
	* 参考[这里](https://jingyan.baidu.com/article/20095761b59d46cb0721b4e9.html)

---

## 快捷键

[官方网页](https://helpx.adobe.com/illustrator/using/default-keyboard-shortcuts.html)

[官方博客](https://blogs.adobe.com/contentcorner/2017/03/17/illustrator-keyboard-shortcuts-a-cheat-sheet/)图片：

![](/assets/AI_cheatsheet.jpeg)