---
layout: post
category: "visualization"
title:  "Lollipop plot"
tags: [plot, visualization]
---


A lollipop(棒糖) plot is an hybrid between a scatter plot and a barplot, which is used to show amino acid mutation along a protein sequence. An example as below:

![img](http://www.cbioportal.org/images/lollipop_example.png)

[lillopop gally]() also show the basic usage of python to achieve this.


```python
# library
 import matplotlib.pyplot as plt
 import numpy as np
 
# create data
 x=range(1,41)
 values=np.random.uniform(size=40)
 
# stem function: first way
 plt.stem(x, values)
 plt.ylim(0, 1.2)
 #plt.show()
 
# stem function: If no X provided, a sequence of numbers is created by python:
 plt.stem(values)
 #plt.show()
 
# stem function: second way
 (markerline, stemlines, baseline) = plt.stem(x, values)
 plt.setp(baseline, visible=False)
 #plt.show()
```

![img](https://python-graph-gallery.com/wp-content/uploads/180_Basic_lolipop_plot.png)