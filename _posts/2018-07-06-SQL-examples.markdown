---
layout: post
category: "linux"
title:  "SQL例子"
tags: [linux, SQL]
---

- TOC
{:toc}

#### 表1 (table_1)

|company|id|age|salary|sex|
|---|---|---|---|---|
|A|001|13|50000|F|
|A|002|25|100000|F|
|B|003|50|200000|M|
|C|004|40|100000|M|
|B|005|40|150000|F|

---

#### 去重计数

```bash
# 罗列不同的id
# distinct: 表全部字段去重，不是部分字段
select distinct id from table_1

# 统计不同id个数
select count(distinct id) from table_1

# 优化版本的统计不同id个数
select count(*) from
(select distinct id from table_1) tb

# count(*): 包括所有列，相当于行数，不忽略值为NULL的
# count(1)：与count(*)一样。
# count(列名)：值包含列名所在列，统计时会忽略NULL
# count时需要看所在列是否可能存在空值NULL

# 例子
CREATE TABLE `score` (
   `id` int(11) NOT NULL AUTO_INCREMENT,
   `sno` int(11) NOT NULL,
   `cno` tinyint(4) NOT NULL,
   `score` tinyint(4) DEFAULT NULL,
   PRIMARY KEY (`id`)
 ) ;
 
A.SELECT sum(score) / count(*) FROM score WHERE cno = 2;
B.SELECT sum(score) / count(id) FROM score WHERE cno = 2;
C.SELECT sum(score) / count(sno) FROM score WHERE cno = 2;
D.SELECT sum(score) / count(score) FROM score WHERE cno = 2;
E.SELECT sum(score) / count(1) FROM score WHERE cno = 2;
F.SELECT avg(score) FROM score WHERE cno = 2;

# ABCE：sum(score)除以行数
# DF：sum(score)除以score不为NULL的行数
# avg(score)：会忽略空值
```

---

#### 聚合函数与group by

```bash
# 聚合函数：基本的数据统计，例如计算最大值、最小值、平均值、总数、求和
# 统计不同性别（F、M）中，不同的id个数
select count(distinct id) from table_1
group by sex

# 统计最大/最小/平均年龄
select max(age), min(age), avg(age) from table_1
group by id
```

* 聚合函数：
	* 作为窗口函数时，与专用窗口函数用法相同，但是函数后括号不能为空，需指定聚合的列名
	* 截止到本行数据，统计数据是多少，或者每一行数据对整体统计数据的影响

---

#### 筛选 where/having

```bash
# 统计A公司的男女人数
select count(distinct id) from table_1
where company = 'A'
group by sex

# 统计各公司的男性平均年龄，并且仅保留平均年龄30岁以上的公司
select company, avg(age) from table_1
where sex = 'M'
group by company
having avg(age) >30
```

---

#### 排序 order by

```bash
# 按年龄全局倒序排序取最年迈的10个人
select id,age from table_1
order by age DESC
limit 10
```

---

#### 条件函数 case when

```bash
# 将salary转换为收入区间进行分组
# case函数格式：
# case when condition1 value1 condition2 value2 ... else NULL end

select id,
(case when CAST(salary as float)<50000 then '0-5万'
when CAST(salary as float)>=50000 and CAST(salary as float)<100000 then '5-10万'
when CAST(salary as float)>=100000 and CAST(salary as float)<200000 then '10-20万'
when CAST(salary as float)>=100000 then '20万以上'
else NULL and from table_1
```

---

#### 窗口函数

* 场景：每组内排名
	* 每个部门按照业绩来排名
	* 找出每个部门排名前N的员工
* 窗口函数：
	* OLAP函数，online analytical processing（联机分析处理）
	* 可对数据库数据进行实时分析处理
* 基本语法：

```bash
<窗口函数> over (partition by <用于分组的列名>
		         order by <用于排序的列名>)
```

* 专用窗口函数：rank、dense_rank、row_number。**注意：函数后面的括号不需要任何参数，保持()空着就行。**
* 聚合函数：sum、avg、count、max、min
* 原则上只能写在select子句中，是对where或者group by子句处理后的结构进行操作

* group by  vs partition by：
	* group by：分组汇总后改变了表的行数，一行只有一个类别
	* partition by：不会改变原表中的函数

* 为什么叫窗口函数？
	* partition by分组后的结果称为窗口，表示范围的意思
	* 功能1：同时具有分组和排序的功能
	* 功能2：不减少原表的函数
	* 语法：<窗口函数> over (partition by 分组列名 order by 排序列名)

---

#### 字符串

##### 拼接 concat

```bash
# 将A和B拼接返回
select concat('www', 'iteblog', 'com') from iteblog
```

---

##### 切分 split

```bash
# 将字符串按照“，”切分，并返回数组
select split("1,2,3", ",") as value_array from table_1

# 切分后赋值
select value_array[0],value_array[1],value_array[2] from (select split("1,2,3", ",") as value_array from table_1) t
```

---

##### 提取子字符串

```bash
#  substr（str,0,len) : 截取从0位开始长度为len的字符串
select substr('abcde', 3, 2) from iteblog # cd
```

---

#### 分组排序 row_number()

```bash
# 按照字段salary倒序排序
select *,row_number() over (order by salary desc) as row_num from table_1

# 按照字段deptid分组后再按照salary倒序编号
select *,row_number() over (partition by deptid order by salary desc) as rank from table_1

# rank：总数不变，排序相同时会重复，会出现1，1，3这种。并列名次会占用下一名次位置。
# dense_rank：总数减小，排序相同时重复，出现1，1，2这种。并列名次不会占用下一名次位置。
# row_number()：排序相同时不重复，会根据顺序排序，不考虑并列名次的情况。 
```

---

#### 根据数值列取top，percentile

```bash
# 获得income字段top10%的阈值
select percentile(CAST(salary as int), 0.9) as income_top10p_threshold from table_1

# 获取income字段的10个百分位点
select percentile(CAST(salary as int), array(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)) as income_top10p_thresholds from table_1

```

---


#### 时间函数

```bash
# 转换为时间格式数据
select to_date('1970-01-01 00:00:00') as start_time from table_1

# 计算数据到当前时间的天数差
select datediff ('2016-12-30','2016-12-29') # 1

# datediff(enddate,stratdate)：计算两个时间的时间差（day)
# date_sub(stratdate, interval 2 day) ：返回开始日期startdate减少days天后的日期
# date_add(startdate,days) ：返回开始日期startdate增加days天后的日期
```

重要的内置MySQL日期函数（[链接](https://www.w3school.com.cn/sql/sql_dates.asp)）：

|函数|功能|
|---|---|
|now()|返回当前的日期和时间|
|curdate()|返回当前的日期|
|curtime()|返回当前的时间|
|date()|提取日期，日期/时间表达式的日期部分|
|extract()|返回日期/时间的单独部分|
|date_add()|给日期添加指定的时间间隔|
|date_sub()|从日期减去指定的时间间隔|
|datediff()|返回两个日期之间的天数|
|date_format()|用不同的格式显示日期/时间|
|day()|取时间字段的天值|
|month()|取时间字段的月值|
|year()|取时间字段的年值|


---

#### drop、delete、truncate

* drop：完全删除表，包括表结构
* delete：只删除表数据，保留表结构，而且可以加where，只删除一行或者多行
* truncate：只删除表数据，保留表结构，不能加where

* 清理表数据的速度，truncate一般比delete更快
* truncate只删除表的数据不删除表的结构

---

#### 练习

```bash
例：有3个表S，C，SC：
S（SNO，SNAME）代表（学号，姓名）
C（CNO，CNAME，CTEACHER）代表（课号，课名，教师）
SC（SNO，CNO，SCGRADE）代表（学号，课号，成绩）

问题：
1. 找出没选过“黎明”老师的所有学生姓名。

select sname from s where SNO not in 
(
select SNO from SC where CNO in 
	(
	select distinct CNO from C where CTEACHER == '黎明'
	)
)

2. 列出2门以上（含2门）不及格学生姓名及平均成绩。

select s.sname, avg_grade from s
join
(select sno from sc where scgrade < 60 group by sno having count(*) >= 2) t1
on s.sno = t1.sno
join
(select sno, avg(scgrade) as avg_grade from sc group by sno ) t2
on s.sno = t2.sno;

3. 既学过1号课程又学过2号课所有学生的姓名。

select SNAME from
(select SNO from SC where CNO = 1) a
join 
(select SNO from SC where CNO = 2) b
on a.sno = b.sno
```

```bash
# table A
Order_id     User_id    Add_time
11701245001 10000    1498882474
11701245002 10001    1498882475

# table B
id     Order_id     goods_id price
1   11701245001    1001     10
2   11701245001    1002     20
3   11701245002    1001     10

购买过goods_id 为1001的用户user_id:
A. select a.user_id from A a, B b where a.order_id = b.order_id and b.goods_id = '1001'
B. select user_id from A where order_id in (select order_id from B where goods_id = '1001')
C. select A.user_id from A left join B on A.order_id = B.order_id and B.goods_id = '1001'
```

---


```bash
# 使用SQL语句建个存储过程proc_stu，然后以student表中的学号Stu_ID为输入参数@s_no，返回学生个人的指定信息

CREATE PROCEDURE [stu].[proc_student]
@s_no AS int
AS
BEGIN
select * from stu.student where Stu_ID=@s_no
END

CREATE PROCEDURE [stu].[proc_student]
@s_no  int
AS
BEGIN
select * from stu.student where Stu_ID=@s_no
END
```

#### 易错题

* 下面哪个字符最可能导致sql注入？A： 单引号，B：/， C：双引号，D：$。 SQL注入的关键是单引号的闭合，因此选单引号。
* select * from user_table where username='xxx' and password='xxx' or '1'='1'，查询到所有用户信息，是单引号导致逻辑发生变化，达到恶意攻击的效果

* SQL主要四部分：
	* 数据定义：DDL，data definition language。定义SQL模式、基本表、视图和索引的创建和撤销操作。
	* 数据操纵：DML，data manipulation language。数据查询和更新。更新分为插入、删除和修改。
	* 数据控制：DCL，data control language。对表和视图的授权，完整性规则的描述，事务控制等。 
	* 事务控制语言：TCL，transaction control language。SQL语句嵌入在宿主语言程序中使用的规则。

* 关系型数据库：
	* 可以表示实体间的1：1，1：n，m：n关系
	* 关系数据模型：使用表格表示实体和实体之间关系的数据模型
	* 可以多对多

* SQL与C语言处理记录的方式是不同的。当将SQL语句嵌入到C语言程序时，为了协调两者而引入：游标。不是堆或者栈或者缓冲区。
	* 游标：系统开设的一个数据缓冲区，存放SQL语句的执行结果集。每个游标有一个名字，用户可以用SQL语句逐一从游标中获取记录，赋值给主变量，交由主语言处理。

* insert into：向表格中插入新的行
* select into：从一个表格中选取数据，然后把数据插入到另一个表中。通常用于创建表的备份或者对记录进行存档。

* 关系数据库事务的基本特征：ACID
	* 原子性：atomic，事务的操作作为整体执行，要么全部执行，要么全部失败
	* 一致性：consistency，数据在事务执行之前和之后，处于一致的状态
	* 隔离性：isolation，多个事务之间是隔离的，互不影响
	* 永久性：durability，一旦事务提交了，对数据库的修改是永久性的

* 数据库及线程发生死锁的原理：
	* 系统资源不足
	* 进程运行推进的顺序不合适
	* 资源分配不当

* where分组前过滤，having分组后过滤，两者不冲突，可同时使用
* having子句必须与groupby子句同时使用，不能单独使用
* group by限定分组条件，对应列进行分组，相同值的被分为一组。当一个sql语句没有groupby时，整张表会自成一组
* 没有聚合函数的使用也可以用having过滤

---

#### 参考

* [SQL \| 数据分析面试必备SQL语句+语法](https://cloud.tencent.com/developer/article/1603982)