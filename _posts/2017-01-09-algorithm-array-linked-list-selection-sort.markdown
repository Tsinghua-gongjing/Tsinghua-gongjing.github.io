---
layout: post
category: "python"
title:  "Algorithm: 数组、链表、选择排序"
tags: [python, scipy]
---

### 内存的工作原理

* 计算机：很多抽屉的集合体，每个抽屉都有地址
* 内存：每一个带有地址的抽屉，用于存放东西，比如：fe0ffeeb是一个内存单元的地址
	* 请求数据存储到地址
	* 计算机分配一个存储地址
	* 若是多项数据：
		* 数组
		* 链表

---

### 数组

![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200306124113.png)

* 例子：待办事项存储
* 数组：在内存中是**相连的**
* 【缺点：添加元素】很麻烦
	* 可预留足够内存：
		* 1）多的内存可能用不上，浪费
		* 2）当超过内存时，还是得进行数据转移，找到一块更大的内存地址
	* 使用链表结构 
* 【优点：读取元素】：
	* 效率很快
	* 链表很慢：比如要读取最后一个元素，比如从第一个开始顺序读取，才能获得最后的元素内存地址

---

### 链表

![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200306124828.png)

* 元素可存储在任何地方
* 每个元素存储了下一个元素的地址，从而使一些列的随机内存地址串在一起
* 类似于寻宝游戏，根据宝物的提示前往下一个地方
* 使用时不需移动元素
* 【优点：添加元素】将新元素放入内存，其地址存到前一个元素中

---

### 中间插入、删除

中间插入：

* 链表：修改前面元素的指向地址即可【更好的选择】
* 数组：后面的元素整体向后移动

删除：

* 链表：修改前面元素的指向地址即可【更好的选择】
* 数组：后面的元素整体向前移动

比较：

* 中间插入：如果内存没有足够空间，则会操作失败
* 删除：总能成功
* 时间复杂度：![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200306125620.png)
* 用哪一个？
	* 取决于具体情况
	* 随机访问：数组的读取方式
	* 顺序访问：链表的读取方式

---

### 选择排序

* 例子：根据歌曲播放次数进行喜好排序
* 方法：**遍历列表，每次找出播放次数最多（最大值）的放在新的列表中**

![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200306130117.png)

* 时间复杂度：O(n x n)
* 问题：需检查的元素越来越少，为什么是O(n x n)？
	* 确实越来越少
	* 第一次n个，后面依次是n-1，n-2，。。。，2和1 => 平均：n/2个
	* 时间：O(n x n/2)
	* 大O表示法：省略常数，比如这里的1/2，因此简写为O(n x n)

---

### python实现

```python
# Finds the smallest value in an array
def findSmallest(arr):
  # Stores the smallest value
  smallest = arr[0]
  # Stores the index of the smallest value
  smallest_index = 0
  for i in range(1, len(arr)):
    if arr[i] < smallest:
      smallest_index = i
      smallest = arr[i]      
  return smallest_index

# Sort array
def selectionSort(arr):
  newArr = []
  for i in range(len(arr)):
      # Finds the smallest element in the array and adds it to the new array
      smallest = findSmallest(arr)
      newArr.append(arr.pop(smallest))
  return newArr

print(selectionSort([5, 3, 6, 2, 10]))
```

---

### 参考

* [图解算法第二章](https://github.com/egonSchiele/grokking_algorithms/blob/master/02_selection_sort/python/01_selection_sort.py)