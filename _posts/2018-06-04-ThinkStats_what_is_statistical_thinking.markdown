---
layout: post
category: "read"
title:  "Think Stats: what's statistical thinking?"
tags: [reading, statistics]
---

Here is the note of reading [Think Stats](https://book.douban.com/subject/24381562/). The Chinese version online is [here](https://jobrest.gitbooks.io/statistical-thinking/content/index.html)

#### 统计思维

1. 概率论：研究随机事件
2. 统计学：根据数据样本推测总体情况
3. 计算：量化分析
4. 经验之谈（anecdotal evidence）：基于个人感受的非公开数据（比如第一个孩子在预产期前出生）。缺点：1）观察的数量太少；2）选择偏差；3）确认偏差；4）不准确。
5. 统计方法：基本步骤，1）收集数据；2）描述性统计；3）探索性数据分析：寻找模式、差异等；4）假设检验：影响的真实性；5）估计。
6. 横断面研究（cross-sectional study）：一群人在某个时间点的情况；纵贯研究（longitudinal study）：在一段时间内反复观察同一群人。采用的数据集全国家庭成长调查（NSFG）属于前者。进行次数：周期（cycle）；参与调查的人称为被调查者（respondent），一组被调查者就称为队列（cohort）。
7. 被调查者文件中的每一行都表示一个被调查者。这行信息称为一条记录（record），组成记录的变量称为字段（field），若干记录的集合就组成了一个表（table）。
8. 观察到的差异称为直观效应（apparent effect），不确定是都一定有意思的事情发生了。

术语：

* 经验之谈（anecdotal evidence） 个人随意收集的证据，而不是通过精心设计并经过研究得到的。
* 直观效应（apparent effect） 表示发生了某种有意思的事情的度量或汇总统计量。
* 人为（artifact） 由于偏差、测量错误或其他错误导致的直观效应。
* 队列（cohort） 一组被调查者。
* 横断面研究（cross-sectional study） 收集群体在特定时间点的数据的研究。
* 字段（field） 数据库中组成记录的变量名称。
* 纵贯研究（longitudinal study） 跟踪群体，随着时间推移对同一组人反复采集数据的研究。
* 过采样（oversampling） 为了避免样本量过少，而增加某个子群体代表的数量。
* 总体（population） 要研究的一组事物，通常是一群人，但这个术语也可用于动物、蔬菜和矿产。
* 原始数据（raw data） 未经或只经过很少的检查、计算或解读而采集和重编码的值。
* 重编码（recode） 通过对原始数据进行计算或是其他逻辑处理得到的值。
* 记录（record） 数据库中关于一个人或其他对象的信息的集合。
* 代表性（representative） 如果人群中的每个成员都有同等的机会进入样本，那么这个样本就具有代表性。
* 被调查者（respondent） 参与调查的人。
* 样本（sample） 总体的一个子集，用于收集数据。
* 统计显著（statistically significant） 若一个直观效应不太可能是由随机因素引起的，就是统计显著的。
* 汇总统计量（summary statistic） 通过计算将一个数据集归结到一个数字（或者是少量的几个数字），而这个数字能表示数据的某些特点。
* 表（table） 数据库中若干记录的集合。