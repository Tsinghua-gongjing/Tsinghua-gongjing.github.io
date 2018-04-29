---
layout: post
category: "visualization"
title:  "Color for plot"
tags: [plot, visualization]
---

# some common used colors

## seaborn

这里选取的颜色主要是来自于seaborn的，感觉这个颜色比较饱和，没有那么鲜艳： 

[![seaborn_colors.png](https://i.loli.net/2018/02/07/5a7ab8d0b9787.png)](https://i.loli.net/2018/02/07/5a7ab8d0b9787.png)


指定获取不同颜色集合中的颜色列表，返回的列表可以用于后续的指定：

~~~ python
import seaborn as sns

def sns_color_ls():
    return sns.color_palette("Set1", n_colors=8, desat=.5)*2
~~~

----------------------------------------

## 参考：
* [seaborn](https://seaborn.pydata.org/index.html): 一个用于统计分析和可视化非常好的包

