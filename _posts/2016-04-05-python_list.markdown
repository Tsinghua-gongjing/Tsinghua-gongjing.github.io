---
layout: post
category: "python"
title:  "Python element list"
tags: [python, list, 列表]
---

## python list

## tricks

#### sort list in zip to keep relation order [stackoverflow](https://stackoverflow.com/questions/9764298/is-it-possible-to-sort-two-listswhich-reference-each-other-in-the-exact-same-w)

```python
>>> list1 = [3,2,4,1, 1]
>>> list2 = ['three', 'two', 'four', 'one', 'one2']
>>> list1, list2 = zip(*sorted(zip(list1, list2)))
>>> list1
(1, 1, 2, 3, 4)
>>> list2 
('one', 'one2', 'two', 'three', 'four')
```

### sort list of str or number

```python
def sort_str_num_ls(ls=[1,2,3]):
    if isinstance(ls[0],int):
        return sorted(ls)
    if isinstance(ls[0],str):
        try:
            return map(str,sorted([int(i) for i in ls]))
        except:
            return sorted(ls)
```

### find index for a value

```python
def find_all_value_index_in_list(lst=[1,2,3,4,5,1],f=1):
    return [i for i, x in enumerate(lst) if x == f]
```

### sum of list of list elements

```python
def list_list_sum(lists=[[1,2],[3,4]],mode='count_sum'):
    if mode == 'count_sum':
        total=sum(sum(ls) for ls in lists)
    if mode == "len_sum":
        total=sum(len(ls) for ls in lists)
    return total
```

### flat nested list

```python
def ls_ls_flat(ls_ls):
    return list(itertools.chain.from_iterable(ls_ls))
```

### convert value list to percentage list

```python
# list to percent list
def list_pct(ls):
    ls=map(float,ls)
    ls_sum=sum(ls)
    ls_pct=[i/ls_sum for i in ls]
    return ls_pct
```

## qucik cheatsheet

source: [Python Crash Course - Cheat Sheets](https://ehmatthes.github.io/pcc/cheatsheets/README.html)

[![beginners_python_cheat_sheet_pcc_lists1.png](https://i.loli.net/2018/04/29/5ae4a62c9d5f0.png)](https://i.loli.net/2018/04/29/5ae4a62c9d5f0.png)

[![beginners_python_cheat_sheet_pcc_lists2.png](https://i.loli.net/2018/04/29/5ae4a62c9651a.png)](https://i.loli.net/2018/04/29/5ae4a62c9651a.png)

