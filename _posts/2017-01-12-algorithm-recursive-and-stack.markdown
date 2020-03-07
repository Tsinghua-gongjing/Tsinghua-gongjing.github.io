---
layout: post
category: "python"
title:  "Algorithm: 递归与栈"
tags: [python, algorithm]
---

### 递归 

* 例子：在一堆盒子中找钥匙
* 不同的解决方案：![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200307170236.png)
* while：只要盒子不为空，则每次取出一个检查
* 递归：
	* 自己调用自己
	* 更清晰，没有性能优势
	* 如果使用循环，程序的性能可能更高；如果使用递归，程序可能更容易理解。如何选择要看什么对你来说更重要。
	* **此时没有盒子堆**，因为栈代替我们这么做了。栈的结构保证我们是否还要检查剩下的盒子堆，以及还有多少需要检查。

* **基线条件**
	* base case
	* 函数不再调用自己，从而避免形成无限循环
* **递归条件**
	* recursive case
	* 函数调用自己

---

### 栈

![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200307170921.png)

* 例子：创建待办事项清单，即一叠便条
* 操作：压入（插入）和弹出（删除并读取）
* 栈：一种简单的数据结构，可用于存储数据，支持压入和弹出的操作
* 调用栈：计算机在内部使用的栈称为调用栈。用于存储多个函数的变量。
* **所有函数调用都进入调用栈**
* 调用栈可能很长，占用大量的内存
* 下面是一个具体的例子，在调用函数的时候，内存的调用情况。可以看到，先调用的入栈，后调用的先被弹出：![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200307172224.png)

---

### 递归调用栈

* 递归函数也是使用调用栈
* 例子：阶乘函数

```python
def fact(x):
	if x == 1:
		return 1
	else:
		return x * fact(x-1)
```

* 下面是在此递归调用过程中，内存栈的变化情况：![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200307172723.png)

---

### 参考

* [图解算法第三章]()

---
