---
layout: post
category: "visualization"
title:  "scatter plot"
tags: [plot, visualization]
---



```
df = sns.load_dataset('tips')
print df.head()

total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4
```

seaborn的pointplot适合画，单一变量在不同类别中的变化趋势，不适合直接展示两个变量之间的相关性：

```
sns.pointplot(x='total_bill', y='tip', data=df)
```

[![scatter_plot1.png](https://i.loli.net/2018/02/25/5a92c24e544f8.png)](https://i.loli.net/2018/02/25/5a92c24e544f8.png)

默认的点与点之间是有连线的，可以指定为没有：

```
sns.pointplot(x="total_bill", y="tip", data=df, linestyles='', )
```

[![scatter_plot1.png](https://i.loli.net/2018/02/25/5a92c2a1a3a42.png)](https://i.loli.net/2018/02/25/5a92c2a1a3a42.png)

所以直接展示两个变量之间的相关性时，可以直接使用pandas的关于df的函数plot：

```
df.plot(kind='scatter', x='total_bill', y='tip')
```

[![scatter_plot1.png](https://i.loli.net/2018/02/25/5a92c34709dd4.png)](https://i.loli.net/2018/02/25/5a92c34709dd4.png)


