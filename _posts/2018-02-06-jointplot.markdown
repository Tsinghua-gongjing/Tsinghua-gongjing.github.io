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

