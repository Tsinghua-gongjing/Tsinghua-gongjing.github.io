---
layout: post
category: "other"
title:  "Adobe Illustrator usage"
tags: [AI, plot]
---

# Adobe Illustrator (AI)

- TOC
{:toc}


## Install:

Refer to [Adobe Illustrator CC 2017 MAC中文破解版](http://www.web3.xin/soft/53.html)

## AI 官方用户指南 (中文版)

##### 简介

* [工作区基础知识](https://helpx.adobe.com/cn/illustrator/using/workspace-basics.html) 

* [创建文档](https://helpx.adobe.com/cn/illustrator/using/create-documents.html)

* [工具](https://helpx.adobe.com/cn/illustrator/using/tools.html)

#### 工作区

* [自定义工作区](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/customizing-workspace.html)

* [“属性”面板](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/properties-panel.html)

* [画板](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/using-multiple-artboards.html)

#### 绘图

* [绘图了解有关 Illustrator 中绘图工具的信息。
绘图基础知识](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/drawing-basics.html)

* [使用钢笔、曲率或铅笔工具绘制](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/drawing-pen-curvature-or-pencil.html)

* [绘制简单线段和形状](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/drawing-simple-lines-shapes.html)

* [绘制像素级优化的图稿](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/pixel-perfect.html)

* [编辑路径](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/editing-paths.html)

* [调整路径段](https://helpx.adobe.com/content/help/cn/zh-Hans/illustrator/using/adjust-path-segments.html)


## 常规操作

* **裁剪图片**：
  - 操作：先在图片上画一个形状（比如长方形、圆圈等），然后同时选择该形状和图片，再`command+7`快捷键，即完成剪切操作。

* **改变点的大小**：
  - 操作：可以改变对象的大小、旋转角度等，比如可以用来改变散点图中点的大小
  - 路径：`object -> transform -> transform each`
  - 快捷键：`shift+option+cmd+d`
  
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

## 快捷键

[官方网页](https://helpx.adobe.com/illustrator/using/default-keyboard-shortcuts.html)

[官方博客](https://blogs.adobe.com/contentcorner/2017/03/17/illustrator-keyboard-shortcuts-a-cheat-sheet/)图片：

![](/assets/AI_cheatsheet.jpeg)