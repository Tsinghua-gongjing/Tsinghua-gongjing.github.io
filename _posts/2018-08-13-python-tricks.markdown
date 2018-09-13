---
layout: post
category: "python"
title:  "Python common used tricks"
tags: [python]
---

- TOC
{:toc}

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

### Sort a column by specific order in a df [stackoverflow](https://stackoverflow.com/questions/23482668/sorting-by-a-custom-list-in-pandas)

```python
# the specific order
sorter = ['a', 'c', 'b']

df['column'] = df['column'].astype("category")
df['column'].cat.set_categories(sorter, inplace=True)

df.sort_values(["column"], inplace=True)
```

### [Free online jupyter](https://jupyter.org/try)

[![online_jupyter_free.jpeg](https://i.loli.net/2018/08/31/5b88bd74ed303.jpeg)](https://i.loli.net/2018/08/31/5b88bd74ed303.jpeg)


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