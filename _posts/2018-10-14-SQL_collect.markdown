---
layout: post
category: "linux"
title:  "SQL常见问题"
tags: [linux, SQL]
---

- TOC
{:toc}

### `left join`时的`on`和`where`

* 参看[SQL中的Join和Where的区别](https://developer.aliyun.com/article/376565)
* left join + on + where
	* 在查询多张表的时候，会生成一张中间表
	* on：在生成临时表的时候使用的条件，不管此条件是否为真，都会返回左表的记录（这是left join的特性所决定的）
	* where：在得到临时表的结果之后，对临时表进行过滤的条件，条件不满足的记录就被过滤掉了
	* 具体参见上述链接页面的例子
