---
layout: post
category: "visualization"
title:  "Heatmap plot"
tags: [plot, visualization]
---

### plot use seaborn

```python
fig,ax=plt.subplots()
sns.heatmap(df, linecolor='grey', linewidths=0.1, cbar=False, square=True, cmap="Greens", ax=ax)
```