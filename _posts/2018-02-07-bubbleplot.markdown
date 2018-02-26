---
layout: post
category: "visualization"
title:  "bubble plot"
tags: [plot, visualization]
---

bubble plot可以结合展示三个变量之间的关系，本文主要是演练了[这里](https://python-graph-gallery.com/271-custom-your-bubble-plot/)这里的代码：

```
# create data
x = np.random.rand(5)
y = np.random.rand(5)
z = np.random.rand(5)
```

---------------------------

控制bubble的颜色、透明度：

```
# Change color with c and alpha
plt.scatter(x, y, s=z*4000, c="red", alpha=0.4)
```

![](https://python-graph-gallery.com/wp-content/uploads/271_Bubble_plot_customization1.png)

---------------------------


控制bubble的形状：

```
plt.scatter(x, y, s=z*4000, marker="D")
```

![](https://python-graph-gallery.com/wp-content/uploads/271_Bubble_plot_customization2.png)

---------------------------


控制bubble的大小:

```
plt.scatter(x, y, s=z*200)
```

![](https://python-graph-gallery.com/wp-content/uploads/271_Bubble_plot_customization3.png)

---------------------------


控制bubble的边缘（线条粗细等）：

```
plt.scatter(x, y, s=z*4000, c="green", alpha=0.4, linewidth=6)
```

![](https://python-graph-gallery.com/wp-content/uploads/271_Bubble_plot_customization4.png)

---------------------------


先导入seaborn，则采用seaborn的主题：

```
import seaborn as sns
plt.scatter(x, y, s=z*4000, c="green", alpha=0.4, linewidth=6)


```

![](https://python-graph-gallery.com/wp-content/uploads/271_Bubble_plot_customization5.png)

---------------------------

同时给气泡上颜色和大小，相当于展示了4个变量：

```
# create data
x = np.random.rand(15)
y = x+np.random.rand(15)
z = x+np.random.rand(15)
z=z*z
 
# Change color with c and alpha. I map the color to the X axis value.
plt.scatter(x, y, s=z*2000, c=x, cmap="Blues", alpha=0.4, edgecolors="grey", linewidth=2)
```

![](https://python-graph-gallery.com/wp-content/uploads/272_Bubble_plot_with_mapped_color.png)