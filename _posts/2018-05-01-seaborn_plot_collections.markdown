---
layout: post
category: "visualization"
title:  "Seaborn plot collections"
tags: [collections, plot]
---


## combine kde plot with regression line

source: [stack overflow](https://stackoverflow.com/questions/48947656/combine-two-seaborn-plots)

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")

rs = np.random.RandomState(5)
mean = [0, 0]
cov = [(1, .5), (.5, 1)]
x1, x2 = rs.multivariate_normal(mean, cov, 500).T
x1 = pd.Series(x1, name="$X_1$")
x2 = pd.Series(x2, name="$X_2$")

g = sns.jointplot(x1, x2, kind="kde", size=7, space=0)
sns.regplot(x1,x2, scatter=False, ax=g.ax_joint)
plt.show()
```

![img](https://i.stack.imgur.com/zzzZx.png)