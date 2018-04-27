---
layout: post
category: "visualization"
title:  "pie plot"
tags: [plot, visualization]
---

# record some of the frequently used tricks

## python plot


* set x axis tick labels

```
# assign tick position and label
# especially for bar plot or time series plot
plt.xticks(range(0, len(compare_cell_ls)), compare_cell_ls, rotation=45)

# Not work: label will off-set 1 (still don't know why)
ax.set_xticklabels(compare_cell_ls, rotation=45)
``` 

* rotate x tick labels

```
# auto get & rotate
ax[0].set_xticklabels(ax[0].xaxis.get_majorticklabels(), rotation=45)
```

* rotate x tick labels in seaborn 

```
# work for last row graph, not Every plots (in FacetGrid)
g.set_xticklabels(rotation=45)
```

* time series plot

```
# data format
tx      start   end     egg     1cell   4cell   64cell  1K      sphere  shield
NM_213431       1497    1503    0.21742857142857144     0.34700000000000003     0.12    0.13285714285714287     0.06685714285714286     0.15614285714285714     0.09571428571428572
NM_199536       1847    1853    0.22228571428571428     0.1551428571428571      0.03528571428571429     0.04671428571428572     0.0008571428571428572   0.045571428571428575    0.060571428571428575
NM_212665       1383    1389    0.12285714285714285     0.07571428571428572     0.027000000000000003    0.026857142857142857    0.019142857142857142    0.02771428571428572     0.024142857142857143
NM_001289906    1900    1906    0.41571428571428576     0.5638571428571428      0.34114285714285714     0.2785714285714286      0.3242857142857143      0.34614285714285714     0.39371428571428574
NM_153663       2491    2497    0.4587142857142856      0.3832857142857143      0.40771428571428575     0.3097142857142857      0.24814285714285714     0.5284285714285715      0.40271428571428575
NM_001098487    1177    1183    0.217   0.2868571428571429      0.13699999999999998     0.14914285714285716     0.11842857142857145     0.22585714285714284     0.19785714285714287
NM_001122844    3696    3702    0.21757142857142855     0.4165714285714285      0.1558571428571429      0.15371428571428572     0.1357142857142857      0.3281428571428572      0.2951428571428571
NM_001083577    2918    2924    0.33399999999999996     0.3514285714285714      0.1827142857142857      0.17557142857142854     0.09285714285714283     0.32899999999999996     0.2894285714285715
NM_001328353    3565    3571    0.32557142857142857     0.3127142857142857      0.19657142857142856     0.2992857142857143      0.241   0.3817142857142857      0.34214285714285714
NM_213231       2183    2189    0.13514285714285715     0.1677142857142857      0.11    0.13785714285714287     0.005   0.14528571428571427     0.043857142857142865

# plot each as trend line
fig,ax=plt.subplots()
for i in df_plot.index:
		ax.plot(range(0, len(col_ls)), df_plot.loc[i, col_ls], color='grey', alpha=0.3, lw=0.3)

# mean value of each state
# axis=0 => mean of each column (add a new row); axis=1 => mean of each row (add a new column)
df_plot_mean = df_plot.loc[:, compare_cell_ls].mean(axis=0)
ax.plot(range(0, len(compare_cell_ls)), df_plot_mean, color='blue')
```

## seaborn plot

* set color list instead of seaborn default

```
color_stages = sns.color_palette('Set1',n_colors=7, desat=0.8)
my_pal = {'egg':color_stages[0], '1cell': color_stages[1], '4cell': color_stages[2], '64cell': color_stages[3], '1K': color_stages[4], 'sphere':color_stages[5], 'shield':color_stages[6]}

```

* set specific color for each cell/category
```
sns.boxplot(x='cell', y='gini', data=df_save_all, ax=ax[0], palette=file_info_dict['my_pal'])
```

