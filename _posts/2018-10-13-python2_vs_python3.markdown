---
layout: post
category: "python"
title:  "Python2 vs Python3"
tags: [plot]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

## 区别列表

| 内容 | Python2 | Python3 |
|:--------|:-------|:--------|
| print   | (1) 语句：print 'abc'  <br> (2) from \_\_future__ import print_function 实现python3的功能| 函数：print('abc')   |
|编码|默认是asscii，经常遇到编码问题|默认是utf-8，不用在抬头指定：# coding=utf-8|
|字符串|(1) str：字节序列 <br> (2) unicode：文本字符串 <br> 两者没有明显的界限 |(1) byte：字节序列（新增） <br> (2) str：字符串<br> 对两者做了严格区分|
|数据类型|(1) long <br> (2) int|int：行为类似于python2中的long|
|除法运算|(1) 除法结果是整除 <br> (2) 1/2结果是0（两个整数相除）<br> (3) 使用from \_\_future__ import division实现该特性|(1) 除法结果是浮点数 <br> (2) 1/2结果是0.5（两个整数相除）|
|输入|(1) input()只接收数字 <br> (2) raw_input()得到的是str类型（为了避免在读取非字符串类型的危险行为）|input()得到str|
|内置函数及方法<br>(filter,map,dict.items等)|大部分返回列表或者元组|返回迭代器对象|
|range|(1) range：返回列表<br> (2) xrange：返回迭代器|range：返回迭代器，取消xrange|
|True/False|(1) 是全局变量，可指向其他对象<br> (2) True对应1，False对应0|(1) 变为两个关键字<br> (2) 永远指向两个固定的对象，不允许再被重新赋值|
|nonlocal|没有办法在嵌套函数中将变量声明为一个非局部变量，只能在函数中声明全局变量|可在嵌套函数中什么非局部变量，使用nonlocal实现|
|类|(1) 默认是旧式类<br> (2) 需要显式继承新式类（object）来创建新式类|(1) 完全移除旧式类<br> (2) 所有类都是新式类，但仍可显式继承object类|
|写文件|print>>SAVEN,'\t'.join()|SAVEFN.write('\t'.join()+'\n')|

---

## 参考

* [Python2 和 Python3 的主要区别——内含思维导图](https://blog.csdn.net/weixin_42105977/article/details/80839700)