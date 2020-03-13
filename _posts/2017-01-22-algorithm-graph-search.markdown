---
layout: post
category: "python"
title:  "Algorithm: 图及图搜索"
tags: [python, algorithm, graph, search]
---

### 目录

- TOC
{:toc}

---

### 图

* 路程问题：
	* 从一个地点出发去往另外一个地方，走哪条路最短？
	* 步骤：
		* 1）使用图模型建立问题模型
		* 2）使用广度优先搜索解决问题
* 最短路径问题：
	* shortest-path problem
	* 找出去往朋友家的最短路径
	* 国际象棋中把对方将死的最少步数
	* 解决算法：**广度优先搜索**
* 图：
	* 由节点+边组成
	* 一个节点可能与众多节点直接相连，这些节点称为**邻居**
	* 模拟一组连接 ![](https://slidesplayer.com/slide/14609437/90/images/2/%E5%9B%BE%E7%9A%84%E5%9F%BA%E6%9C%AC%E6%A6%82%E5%BF%B5+%E6%9C%89%E5%90%91%E5%9B%BE+G1+%E6%97%A0%E5%90%91%E5%9B%BE+G2+A+B+A+B+E+C+D+C+D+%E7%BB%93%E7%82%B9%E6%88%96+%E9%A1%B6%E7%82%B9%EF%BC%9A+A+B+%E7%BB%93%E7%82%B9%E6%88%96+%E9%A1%B6%E7%82%B9%EF%BC%9A+A+B.jpg)
	
---

### 广度优先搜索

* 回答两类问题：
	* （1）从节点A出发，有前往节点B的路径吗？
	* （2）从节点A出发，前往节点B的哪条路径最短？
* 例子：在朋友圈找到芒果经销商【第一类问题，有没有】
	* 先在自己的朋友里面找，看有没有
	* 如果没有的话，把该朋友的朋友也加入到搜索队列的后面，因为后面还要接着搜索 ![graph_example.png](https://i.loli.net/2020/03/08/6cT3WqAeV7SOiDN.png)

---

#### 查找最短路径

* 上述的问题中：
	* 一度关系胜过二度关系，以此类推
	* 先搜索一度关系，再往外延伸
	* 因此一度关系在二度关系之前加入查找名单
* 比如顺序添加，先添加先检查
* 数据结构：队列
	* queue
	* 按添加顺序进行检查

---

#### 队列

* 原理：与等公交类似，先到先上车
* 操作：
	* 不能随机访问队列中元素
	* 入队、出队
	* 队列：先进先出，FIFO（first in first out）
	* 栈：后进先出，LIFO（last in first out）![queue_vs_stack.png](https://i.loli.net/2020/03/08/VnSdozew4vxqTu7.png)

---

#### 图实现

* 有向图：directed graph，关系是单向的
* 无向图：undirected graph，没有箭头，直接互连的节点为邻居 
* **散列表实现图：可表示相互之间的归属关系**，这个只是存储彼此关系用于查找的，不是搜索的关键，**搜索的关键是队列**
* 算法实现：
	* 创建**队列**，存储要检查的人
	* 弹出一个人，检查是否为经销商
		* 是：成功
		* 否：将这个人的朋友加入到队列
	* 重复弹出检查
	* 如果队列为空（全部都检查完），就说明人际关系网中没有经销商 ![graph_search_BSF.png](https://i.loli.net/2020/03/08/vXxbriZwkl3IfHq.png)
* 注意：
	* 搜索的停止条件，1）找到一位经销商，2）队列变为空，即找不到经销商
	* 如果某人是两人的朋友，可能需要检查两次
		* 标记某人是否检查过，可以使用列表进行记录
		* 没有检查过则进行检测
		* 否则容易进入死循环。比如你的朋友，其朋友只有你，那么从自己出发，就只会搜索你和他两人。

```python
from collections import deque

# 判断是否为经销商
def person_is_seller(name):
      return name[-1] == 'm'

# 构建关系图，存储彼此之间的关系
graph = {}
graph["you"] = ["alice", "bob", "claire"]
graph["bob"] = ["anuj", "peggy"]
graph["alice"] = ["peggy"]
graph["claire"] = ["thom", "jonny"]
graph["anuj"] = []
graph["peggy"] = []
graph["thom"] = []
graph["jonny"] = []
# 注意这里用的是列表存储朋友
# 即使没有朋友，也需要指定值为空列表
# 因为后面的seach是列表，search列表更新时需要支持直接相加的操作

def search(name):
    search_queue = deque()
    search_queue += graph[name]
    # This array is how you keep track of which people you've searched before.
    searched = []
    while search_queue:
        person = search_queue.popleft()
        # Only search this person if you haven't already searched them.
        if person not in searched:
            if person_is_seller(person):
                print(person + " is a mango seller!")
                return True
            else:
                search_queue += graph[person]
                # Marks this person as searched
                searched.append(person)
    return False

search("you")
```

---

#### 运行时间

* 沿着边搜索，至少运行O(边数)
* 队列存储检查的人，所以这部分为O(人数)
* 广度优先搜索：O(人数+边数) =》 O(V+E), V:定点数，E:边数

---

### 参考

* [图解算法第六章]()

---
