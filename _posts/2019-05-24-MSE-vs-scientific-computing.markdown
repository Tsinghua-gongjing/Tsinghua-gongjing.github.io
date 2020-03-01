---
layout: post
category: "statistics"
title:  "Basic operations on matrix"
tags: [reading, statistics]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 使用numpy解析线性方程式

例子：

![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20200301094910.png)

```python
import numpy as np

A = np.array([[2, 1, 1], [1, 3, 2], [1, 0, 0]]) 

B = np.array([4, 5, 6]) 

# linalg.solve is the function of NumPy to solve a system of linear scalar equations
print ("Solutions:\n",np.linalg.solve(A, B ) )

Solutions:
[  6.  15. -23.]
```

---