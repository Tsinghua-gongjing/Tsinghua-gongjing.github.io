---
layout: post
category: "linux"
title:  "SQL例子"
tags: [linux, SQL]
---

- TOC
{:toc}

### 表1 (table_1)

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
