---
layout: post
category: "python"
title:  "Python common used tricks"
tags: [python]
---

- TOC
{:toc}

---

### Calculate distance between two intervals [stackoverflow](https://stackoverflow.com/questions/16843409/finding-integer-distance-between-two-intervals)

```python
def solve(r1, r2):
     # sort the two ranges such that the range with smaller first element
     # is assigned to x and the bigger one is assigned to y
     x, y = sorted((r1, r2))

     #now if x[1] lies between x[0] and y[0](x[1] != y[0] but can be equal to x[0])
     #then the ranges are not overlapping and return the differnce of y[0] and x[1]
     #otherwise return 0 
     if x[0] <= x[1] < y[0] and all( y[0] <= y[1] for y in (r1,r2)):
        return y[0] - x[1]
     return 0
... 
>>> solve([0,10],[12,20])
2
>>> solve([5,10],[1,5])
0
>>> solve([5,10],[1,4])
1
```

---

### Sort a column by specific order in a df [stackoverflow](https://stackoverflow.com/questions/23482668/sorting-by-a-custom-list-in-pandas)

```python
# the specific order
sorter = ['a', 'c', 'b']

df['column'] = df['column'].astype("category")
df['column'].cat.set_categories(sorter, inplace=True)

df.sort_values(["column"], inplace=True)
```

---

### [Free online jupyter](https://jupyter.org/try)

[![online_jupyter_free.jpeg](https://i.loli.net/2018/08/31/5b88bd74ed303.jpeg)](https://i.loli.net/2018/08/31/5b88bd74ed303.jpeg)

A free Python/R notebook can also be created online at [https://rdrr.io/](https://rdrr.io/).

---

### Normalize data frame by row/column sum [stackoverflow](https://stackoverflow.com/questions/35678874/normalize-rows-of-pandas-data-frame-by-their-sums/35679163)

```python
t = pd.DataFrame({1:[1,2,3], 2:[3,4,5], 3:[6,7,8]})
t

	1	2	3
0	1	3	6
1	2	4	7
2	3	5	8
```

```python
# by row sum
t.div(t.sum(axis=1), axis=0)

	1	2	3
0	0.100000	0.300000	0.600000
1	0.153846	0.307692	0.538462
2	0.187500	0.312500	0.500000

# by column sum
t.div(t.sum(axis=0), axis=1)

	1	2	3
0	0.166667	0.250000	0.285714
1	0.333333	0.333333	0.333333
2	0.500000	0.416667	0.380952
```

---

### Write multiple df to one sheet with multiple tab [stackoverflow](https://stackoverflow.com/questions/32957441/putting-many-python-pandas-dataframes-to-one-excel-worksheet)


```python
def dfs_tabs(df_list, sheet_list, file_name):
    writer = pd.ExcelWriter(file_name,engine='xlsxwriter')   
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet, startrow=0 , startcol=0, index=False)   
    writer.save()
```

---

### 根据df某一列的值，找到其最大值所对应的index

As discussed [here](https://stackoverflow.com/questions/39964558/pandas-max-value-index):

```python
df = pd.DataFrame({'favcount':[1,2,3], 'sn':['a','b','c']})

print (df)
   favcount sn
0         1  a
1         2  b
2         3  c

print (df.favcount.idxmax())
2

print (df.ix[df.favcount.idxmax()])
favcount    3
sn          c
Name: 2, dtype: object

print (df.ix[df.favcount.idxmax(), 'sn'])
c
```

---

### 设置画图时使用Helvetica字体

主要参考这篇文章：[Changing the sans-serif font to Helvetica](http://fowlerlab.org/2019/01/03/changing-the-sans-serif-font-to-helvetica/)。转换好的字体文件放在了[这里](https://github.com/Tsinghua-gongjing/blog_codes/tree/master/files/font)，可下载使用。

```bash
# 在mac上找到Helvetica字体
$ ls /System/Library/Fonts/Helvetica.ttc

# 复制到其他的位置
$ cp /System/Library/Fonts/Helvetica.ttc ~/Desktop

# 使用online的工具转换为.tff文件
# 这里使用的是: https://www.files-conversion.com/font/ttc

# 定位python库的字体文件
$ python -c 'import matplotlib ; print(matplotlib.matplotlib_fname())'

/Users/gongjing/usr/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc

# 将tff文件放到上述路径的font目录下
$ cp Helvetica.ttf /Users/gongjing/usr/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf

# 修改matplotlibrc文件
#font.sans-serif : DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Helvetica, Lucid, Arial, Avant Garde, sans-serif

=》font.sans-serif : Helvetica, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Avant Garde, sans-serif

# 重启jupyter即可
```

上面是设置全局的，也可以显示的在代码中指定，可以参考[这里](https://stackoverflow.com/questions/21321670/how-to-change-fonts-in-matplotlib-python)：

```python
# 显示指定在此脚本中用某个字体
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Helvetica"

# 对于不同的部分（标题、刻度等）指定不同的字体
csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Helvetica'}
plt.title('title',**csfont)
plt.xlabel('xlabel', **hfont)
plt.show()
```

---

### 画图时支持中文

画图时，如果需要使用中文label，需要设置，主要参考[这里](https://www.zhihu.com/question/25404709)：

* 1、下载`SimHei`黑体字体[文件](https://link.zhihu.com/?target=http%3A//www.fontpalace.com/font-details/SimHei/)
* 2、将下载的`.tff`文件放到`matplotlib`包的路径下，路径为：`matplotlib/mpl-data/fonts/ttf`，可以使用`pip show matplotlib`查看包安装的位置
* 3、修改配置文件：`matplotlibrc`，一般在`matplotlib/mpl-data/`这个下面。
	* font.family : sans-serif
	* font.sans-serif : SimHei, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif
	* axes.unicode_minus:False
* 4、删除`/Users/gongjing/.matplotlib`下面缓存的字体文件
* 5、再直接调用即可`plt.rcParams["font.family"] = "SimHei"`

---

### 将array里面的NA值替换为其他实数值

可以使用numpy的函数[nan\_to\_num](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.nan_to_num.html)：`numpy.nan_to_num(x, copy=True)]`

```python
x = np.array([np.inf, -np.inf, np.nan, -128, 128])

# 默认copy=True，不改变原来数组的值
np.nan_to_num(x)
array([  1.79769313e+308,  -1.79769313e+308,   0.00000000e+000,
        -1.28000000e+002,   1.28000000e+002])
        
# 设置copy=False，原来数组的值会被替换
np.nan_to_num(x, copy=False)
```

---

### 核查文件夹是否存在否则创建

As discussed [here](https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory):

```python
import os
def check_dir_or_make(d):
    if not os.path.exists(d):
        os.makedirs(d)
```

---

### 把df中的某一列按照特定字符分隔成多列

As discussed [here](https://cmdlinetips.com/2018/11/how-to-split-a-text-column-in-pandas/):

```python
df = pd.DataFrame({'Name': ['Steve_Smith', 'Joe_Nadal', 
                           'Roger_Federer'],
                 'Age':[32,34,36]})
                 
# Age	Name
# 0	32	Steve_Smith
# 1	34	Joe_Nadal
# 2	36	Roger_Federer

df[['First','Last']] = df.Name.str.split("_",expand=True,)
# expand需要设置为True，负责报错说原来df没有“first”，“last”列

# Age	Name	First	Last
# 0	32	Steve_Smith	Steve	Smith
# 1	34	Joe_Nadal	Joe	Nadal
# 2	36	Roger_Federer	Roger	Federer
```

----