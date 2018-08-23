---
layout: post
category: "visualization"
title:  "Frequently used ticks"
tags: [plot, visualization]
---

# record some of the frequently used tricks

## python plot


* set x axis tick labels

```python
# assign tick position and label
# especially for bar plot or time series plot
plt.xticks(range(0, len(compare_cell_ls)), compare_cell_ls, rotation=45)

# Not work: label will off-set 1 (still don't know why)
ax.set_xticklabels(compare_cell_ls, rotation=45)
``` 

* rotate x tick labels

```python
# auto get & rotate
ax[0].set_xticklabels(ax[0].xaxis.get_majorticklabels(), rotation=45)
```

* rotate x tick labels in seaborn 

```python
# work for last row graph, not Every plots (in FacetGrid)
g.set_xticklabels(rotation=45)
```

* time series plot

```python
# data format
# each row denote a gene's expression under different condition
[zhangqf7@loginview02 HuR]$ head predict_RBP_binding_combine.compare.txt|cut -f 4-7
egg     1cell   4cell   64cell
0.21742857142857144     0.34700000000000003     0.12    0.13285714285714287
0.22228571428571428     0.1551428571428571      0.03528571428571429     0.04671428571428572
0.12285714285714285     0.07571428571428572     0.027000000000000003    0.026857142857142857
0.41571428571428576     0.5638571428571428      0.34114285714285714     0.2785714285714286
0.4587142857142856      0.3832857142857143      0.40771428571428575     0.3097142857142857
0.217   0.2868571428571429      0.13699999999999998     0.14914285714285716
0.21757142857142855     0.4165714285714285      0.1558571428571429      0.15371428571428572
0.33399999999999996     0.3514285714285714      0.1827142857142857      0.17557142857142854
0.32557142857142857     0.3127142857142857      0.19657142857142856     0.2992857142857143

# plot each as trend line
fig,ax=plt.subplots()
for i in df_plot.index:
		ax.plot(range(0, len(col_ls)), df_plot.loc[i, col_ls], color='grey', alpha=0.3, lw=0.3)

# mean value of each state
# axis=0 => mean of each column (add a new row); axis=1 => mean of each row (add a new column)
df_plot_mean = df_plot.loc[:, compare_cell_ls].mean(axis=0)
ax.plot(range(0, len(compare_cell_ls)), df_plot_mean, color='blue')
```

* remove legend (also work in seaborn)

```python
ax.legend_.remove()
```

* set equal axis and x_lim/ylim [github: set_ylim not working with plt.axis('equal') ](https://github.com/matplotlib/matplotlib/issues/8093)

```python
plt.plot((.1, .3))
ax.axis('square')
ax.set_xlim(0.1, 0.3)
```

* remove spines on the right and top

```python
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

* annotate point/position with text [stackoverflow](https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point)

```python
y = [2.56422, 3.77284, 3.52623, 3.51468, 3.02199]
z = [0.15, 0.3, 0.45, 0.6, 0.75]
n = [58, 651, 393, 203, 123]

fig, ax = plt.subplots()
ax.scatter(z, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]))
```

## seaborn plot

* set color list instead of seaborn default

```python
color_stages = sns.color_palette('Set1',n_colors=7, desat=0.8)
my_pal = {'egg':color_stages[0], '1cell': color_stages[1], '4cell': color_stages[2], '64cell': color_stages[3], '1K': color_stages[4], 'sphere':color_stages[5], 'shield':color_stages[6]}

```

* set specific color for each cell/category

```python
sns.boxplot(x='cell', y='gini', data=df_save_all, ax=ax[0], palette=file_info_dict['my_pal'])
```

* plot multiple heatmap

```python

### 指定height_ratios，一般根据每个集合具有的feature的数目
fig, ax = plt.subplots(3,1,figsize=(32, 25), gridspec_kw = {'height_ratios':[19, 15, 7]}, sharey=False, sharex=True)

### plot heatmap
h1 = sns.heatmap(df_plot_all[function_ls[1:]].T,linecolor='grey', linewidths=0.1, cbar=False, square=True, cmap="Greens", ax=ax[0])
h2 = sns.heatmap(df_plot_all[localization_ls].T,linecolor='grey', linewidths=0.1, cbar=False, square=True, cmap="Greens", ax=ax[1])
h3 = sns.heatmap(df_plot_all[domain_ls].T,linecolor='grey', linewidths=0.1, cbar=False, square=True, cmap="Greens", ax=ax[2])

### keep one xlabel for all, also keep yticklabels
# h1.set(xlabel='', ylabel='Heatmap1')
# h2.set(xlabel='', ylabel='Heatmap2')
# h3.set(xlabel='Columns', ylabel='Heatmap3')

### keep one xlabel for all, remove yticklabels
h1.set(xlabel='', ylabel='Heatmap1', yticks=[])
h2.set(xlabel='', ylabel='Heatmap2', yticks=[])
h3.set(xlabel='Columns', ylabel='Heatmap3', yticks=[], xticks=[])

### set yticklabels on the right
ax[0].yaxis.tick_right()
ax[0].set_yticklabels(ax[0].yaxis.get_majorticklabels(), rotation=0)

ax[1].yaxis.tick_right()
ax[1].set_yticklabels(ax[1].yaxis.get_majorticklabels(), rotation=0)

ax[2].yaxis.tick_right()
ax[2].set_yticklabels(ax[2].yaxis.get_majorticklabels(), rotation=0)
ax[2].set_xticklabels(ax[2].xaxis.get_majorticklabels(), rotation=90)

plt.tight_layout()
plt.savefig('./test.png')
plt.close()
```

[![multuple_heatmap](https://i.loli.net/2018/08/23/5b7e1fbbc40c1.png)](https://i.loli.net/2018/08/23/5b7e1fbbc40c1.png)


## inkscape

* convert pdf to svg ([stackoverflow](https://stackoverflow.com/questions/4120567/convert-pdf-to-svg)), only for first page

```bash
/Applications/Inkscape.app/Contents/Resources/bin/inkscape -l Python_graph.svg Python_graph.pdf
```

## MagicImage

* conbine multiple image into one figure/pdf file

```bash
# auto rows and columns
montage *png out.pdf

# use filename to label each image
montage -label '%f' * out.pdf

# 4 columns x multiple rows
montage *.png -mode concatenate -tile 4x out.pdf
```