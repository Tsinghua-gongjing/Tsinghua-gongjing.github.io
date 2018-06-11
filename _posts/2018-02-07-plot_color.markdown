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

<table border = "1">
         <tr>
            <th>category</th>
            <th bgcolor = "#4C72B0">blue</th>
            <th bgcolor = "#55A868">green</th>
            <th bgcolor = "#C44E52">red</th>
            <th bgcolor = "#8172B2">purple</th>
            <th bgcolor = "#CCB974">orange</th>
            <th bgcolor = "#64B5CD">cyan</th>
         </tr>
         <tr>
            <td>red</td>
            <td>74</td>
            <td>83</td>
            <td>202</td>
            <td>129</td>
            <td>205</td>
            <td>98</td>
         </tr>
         <tr>
            <td>green</td>
            <td>113</td>
            <td>169</td>
            <td>75</td>
            <td>112</td>
            <td>185</td>
            <td>180</td>
         </tr>
         <tr>
         	  <td>blue</td>
            <td>178</td>
            <td>102</td>
            <td>78</td>
            <td>182</td>
            <td>111</td>
            <td>208</td>
         </tr>
         <tr>
            <td>HEX1</td>
            <td>#4C72B0</td>
            <td>#55A868</td>
            <td>#C44E52</td>
            <td>#8172B2</td>
            <td>#CCB974</td>
            <td>#64B5CD</td>
         </tr>
         <tr>
            <td>RGB</td>
            <td>74,113,178</td>
            <td>83,169,102</td>
            <td>202,75,78</td>
            <td>129,112,182</td>
            <td>205,185,111</td>
            <td>98,180,208</td>
         </tr>
</table>


指定获取不同颜色集合中的颜色列表，返回的列表可以用于后续的指定：

~~~ python
import seaborn as sns

def sns_color_ls():
    return sns.color_palette("Set1", n_colors=8, desat=.5)*2
    
sns_color_ls = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
~~~

----------------------------------------

## 参考：
* [seaborn](https://seaborn.pydata.org/index.html): 一个用于统计分析和可视化非常好的包

