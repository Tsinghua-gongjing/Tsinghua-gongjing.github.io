---
layout: post
category: "python"
title:  "Algorithm: 时间复杂度"
tags: [python, algorithm]
---

- TOC
{:toc}

---

## 时间复杂度

* 是一个函数，定性描述算法算法的运行时间
* 大O表示法：
	* 算法的操作数（Operation）
	* 指出了算法运行时间的增速
	* 单位不是秒
	* **指出最糟糕情况下的运行时间**
* 常见大O运行时间：
	* O(logn): 对数时间，比如二分查找
	* O(n): 线性时间，比如简单(穷举)查找
	* O(n * logn): 比如快速排序（较快的排序算法）
	* O(n^2): 比如选择排序（很慢的排序算法）
	* O(n!): 比如旅行商问题，也是很慢的算法
	* 快 O(1)<O(㏒2n)<O(n)<O(n2)<O(2n) 慢

	* 常见函数值与n的关系，可以看到，当n增大时，阶乘(n!)、平方(n^2)等是操作数增长最快的，也是最慢的 ![time_complexity_function.png](https://i.loli.net/2020/03/05/ocs9eZVSKRt8Qg6.png)

---

## 常见算法的复杂度

### 搜索

![time_complexity_search.png](https://i.loli.net/2020/03/05/5CkgHMhfKAlNLBX.png)

---

### 排序

![time_complexity_sort.png](https://i.loli.net/2020/03/05/wv2hJAgmU1X4YpH.png)

---

### 数据结构

![time_complexity_data_structure.png](https://i.loli.net/2020/03/05/utMx81zybJOHnNR.png)

---

### 堆

![time_complexity_heap.png](https://i.loli.net/2020/03/05/RWBybCjJfpNe1z3.png)

---

### 图

![time_complexity_graph.png](https://i.loli.net/2020/03/05/EyFteR6aqOrgkM9.png)

---

## 参考

* [算法图解第一章](https://github.com/egonSchiele/grokking_algorithms/tree/master/01_introduction_to_algorithms/python)
* [常用算法时间复杂度](https://blog.csdn.net/l975764577/article/details/39399077)