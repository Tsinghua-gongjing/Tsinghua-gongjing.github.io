---
layout: post
category: "visualization"
title:  "Joint plot"
tags: [plot, visualization]
---

### plot use `sns.jointplot`

```python
tips = sns.load_dataset("tips")
g = sns.jointplot(x="total_bill", y="tip", data=tips)
```

![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20190726223744.png)