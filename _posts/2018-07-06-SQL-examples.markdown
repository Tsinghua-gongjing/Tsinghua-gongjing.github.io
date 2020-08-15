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

# rank：总数不变，排序相同时会重复，会出现1，1，3这种
# dense_rank：总数减小，排序相同时重复，出现1，1，2这种
# row_number()：排序相同时不重复，会根据顺序排序 
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
# date_sub(stratdate,days) ：返回开始日期startdate减少days天后的日期
# date_add(startdate,days) ：返回开始日期startdate增加days天后的日期
```

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


---

#### 参考

* [SQL \| 数据分析面试必备SQL语句+语法](https://cloud.tencent.com/developer/article/1603982)