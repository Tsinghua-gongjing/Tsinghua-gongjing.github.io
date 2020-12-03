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

---

### 查询语句执行顺序

需要遵循以下顺序：

|子句|说明|是否必须使用|
|---|---|---|
|SELECT|要返回的列或者表达式|是|
|FROM|从中检索数据的表|仅在从表中选择数据时使用|
|WHERE|行级过滤|否|
|GROUP BY|分组说明|仅在按组计算聚集时使用|
|HAVING|组级过滤|否|
|ORDER BY|输出排序顺序|否|
|LIMIT|要检索的行数|否|

---

### `LIMIT`的用法

* 用法：`select * from tableName limit i,n`
* `i`：查询结果的索引值，默认从n开始
* `n`：查询结果返回的数量

```sql
--检索前10行数据，显示1-10条数据
select * from Customer LIMIT 10;

--检索从第2行开始，累加10条id记录，共显示id为2....11
select * from Customer LIMIT 1,10;

--检索从第6行开始向前加10条数据，共显示id为6,7....15
select * from Customer limit 5,10;

--检索从第7行开始向前加10条记录，显示id为7,8...16
select * from Customer limit 6,10;
```

---

### `OFFSET`的用法

* 用法：`OFFSET n`
* `n`：表示跳过n个记录
* 注意：当 limit和offset组合使用的时候，limit后面只能有一个参数，表示要取的的数量，offset表示要跳过的数量。

```sql
-- 跳过第一个记录
select * from article OFFSET 1

-- 跳过第一个记录，提取接下来的3个记录
-- 当LIMIT和OFFSET联合使用的时候，limit后面只能有一个参数
select * from article LIMIT 3 OFFSET 1

-- 跳过第一个记录，从index=1开始，提取接下来的3个记录
select* from article LIMIT 1,3
```

---

### `case...when...`用法

条件表达式函数：

```sql
CASE WHEN condition THEN result
 
[WHEN...THEN...]
 
ELSE result
 
END
```

例子：

```sql
SELECT
    STUDENT_NAME,
    (CASE WHEN score < 60 THEN '不及格'
        WHEN score >= 60 AND score < 80 THEN '及格' ---不能连续写成 60<=score<80
        WHEN score >= 80 THEN '优秀'
        ELSE '异常' END) AS REMARK
FROM
    TABLE
```

---

### 计算每个类别的数目及占比

参考这里：[Percentage from Total SUM after GROUP BY SQL Server](https://stackoverflow.com/questions/46909494/percentage-from-total-sum-after-group-by-sql-server)：

```sql
--- count based
--- 要加和的是总entry数目
SELECT
 t.device_model,
 COUNT(t.device_model) AS num,
 COUNT(t.device_model)/SUM(COUNT(t.device_model)) OVER () AS Percentage
 
 --- sum based
 --- 要加和是另外一列
 SELECT P.PersonID, SUM(PA.Total),
       SUM(PA.Total) * 100.0 / SUM(SUM(PA.Total)) OVER () AS Percentage
```

---

### hive中时间与字符串的转换

参考[这里](https://blog.csdn.net/u013421629/article/details/80068090)，注意不用于直接sql中的函数:

```sql
--方法1: from_unixtime+ unix_timestamp
--20171205转成2017-12-05 
select from_unixtime(unix_timestamp('20171205','yyyymmdd'),'yyyy-mm-dd') from dual;

--2017-12-05转成20171205
select from_unixtime(unix_timestamp('2017-12-05','yyyy-mm-dd'),'yyyymmdd') from dual;

--方法2: substr + concat
--20171205转成2017-12-05 
select concat(substr('20171205',1,4),'-',substr('20171205',5,2),'-',substr('20171205',7,2)) from dual;

--2017-12-05转成20171205
select concat(substr('2017-12-05',1,4),substr('2017-12-05',6,2),substr('2017-12-05',9,2)) from dual;
```

---

### `with as`用法

* 子查询部分（subquery factoring），是用来定义一个SQL片断，该SQL片断会被整个SQL语句所用到。这个语句算是公用表表达式（CTE）
* 尤其是多表查询，后面跟随多个join时，比如先确定需要查询哪些用户或者设备
* 支持多个子查询

```sql
--后面就可以直接把query_name1，query_name2当做表格来用了
WITH query_name1 AS (
     SELECT ...
     )
   , query_name2 AS (
     SELECT ...
       FROM query_name1
        ...
     )
SELECT ...
```

---

### SQL书写规范

* 所有表名、字段名全部小写，系统保留字、内置函数名、Sql保留字大写
* 对较为复杂的sql语句加上注释
* 注释单独成行、放在语句前面，可采用单行/多行注释。（-- 或 /* */ 方式）
* where子句书写时，每个条件占一行，语句令起一行时，以保留字或者连接符开始，连接符右对齐
* 多表连接时，使用表的别名来引用列

参考：[SQL书写规范](https://www.cnblogs.com/yangkunlisi/archive/2011/09/14/2176773.html)

---

### hive使用变量

```sql
-- 获取1-2号的新设备
-- SET时不用添加引号，使用时添加引号
SET date1 = 20180701;
SET date2 = 20180702;
SET app = app_name;

WITH new_did AS (
  SELECT
    p_date,
    device_id
  FROM
    your_table
  WHERE
    p_date BETWEEN '${hiveconf:pdate1}'
    AND '${hiveconf:pdate2}'
    AND product = '${hiveconf:app}'
    AND is_today_new = 1
)
```