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

---

### 相同的id根据其他列的值选择最大或者最小的

* 参考[SQL 多条数据中取最大值？](https://bbs.csdn.net/topics/392295893)

```sql
# 表格

ID     Uname  Price   BuyDate
1      张三        180     2017-12-1
2      张三        280     2017-12-7
3      李四        480     2017-12-10
4      李四        280     2017-12-11
5      王武        280     2017-12-1
6      王武        880     2017-12-11
7      王武        380     2017-12-15

# 想要
ID     Uname  Price   BuyDate
2      张三        280     2017-12-7
3      李四        480     2017-12-10
6      王武        880     2017-12-11

# 语句
SELECT * FROM (
    SELECT *,ROW_NUMBER()OVER(PARTITION BY Uname ORDER BY Price DESC ) AS rn FROM #t
) AS t WHERE t.rn=1

# 使用降序或者升序配合n=1，可以选择最小或者最大的
# 配合类似n<=5，可以选择最小或者最大的5个
```