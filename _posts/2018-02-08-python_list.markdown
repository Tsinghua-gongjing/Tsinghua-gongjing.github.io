---
layout: post
category: "python"
title:  "Python element list"
tags: [python, list, 列表]
---

## python list

### tricks

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


### qucik cheatsheet

source: [Python Crash Course - Cheat Sheets](https://ehmatthes.github.io/pcc/cheatsheets/README.html)

[![beginners_python_cheat_sheet_pcc_lists1.png](https://i.loli.net/2018/04/29/5ae4a62c9d5f0.png)](https://i.loli.net/2018/04/29/5ae4a62c9d5f0.png)

[![beginners_python_cheat_sheet_pcc_lists2.png](https://i.loli.net/2018/04/29/5ae4a62c9651a.png)](https://i.loli.net/2018/04/29/5ae4a62c9651a.png)

