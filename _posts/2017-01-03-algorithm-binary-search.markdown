---
layout: post
category: "python"
title:  "Algorithm: 穷举、二分查找"
tags: [python, algorithm, search]
---

## 问题背景

查找问题：

* 在电话薄中找名字以K打头的人
* 在字典中找一个以O打头的单词
* facebook核查用户登录的信息

---

### 简单（穷举）查找

每次查找排除一个数字，但是这个非常耗时。下面是一个动画，演示寻找数字37：

![](https://i.pinimg.com/originals/e2/9a/31/e29a31c78bcc0d07c612adc77acc09a0.gif)

---

### 二分查找

描述：输入一个**有序的**元素列表，如果查找的元素包含在列表中，返回其位置，否则返回null。

思想：每次都通过中间值进行判断，从而否定掉一半的可能性

* 注意：必须是有序的
* 时间复杂度：**对于含有n个元素的列表，最多需要log2(n)步，而简单查找最多需要n步**

```python
def binary_search(list, item):
  # low and high keep track of which part of the list you'll search in.
  low = 0
  high = len(list) - 1

  # While you haven't narrowed it down to one element ...
  while low <= high:
    # ... check the middle element
    mid = (low + high) // 2
    guess = list[mid]
    # Found the item.
    if guess == item:
      return mid
    # The guess was too high.
    if guess > item:
      high = mid - 1
    # The guess was too low.
    else:
      low = mid + 1

  # Item doesn't exist
  return None

my_list = [1, 3, 5, 7, 9]
print(binary_search(my_list, 3)) # => 1

# 'None' means nil in Python. We use to indicate that the item wasn't found.
print(binary_search(my_list, -1)) # => None
```

---

### 参考

* [算法图解第一章](https://github.com/egonSchiele/grokking_algorithms/tree/master/01_introduction_to_algorithms/python)