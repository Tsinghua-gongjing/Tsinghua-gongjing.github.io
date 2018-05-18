---
layout: post
category: "genomics"
title:  "Circos tableview tool to visualize interaction correlation or distribution"
tags: [genomics, circos]
---

### [circos tableview](http://mkweb.bcgsc.ca/tableviewer/)

Circos can not only show the connections between different elements, e.g., genomic regions, RNA molecules and so on. Another userful too tablevier can be used to display the percentage of each crosslinks among multiple relationships. In [RISE database](http://rise.life.tsinghua.edu.cn/statistics.html) we collected RNA-RNA interactions (RRIs) from various sources, and [tableviewer](http://mkweb.bcgsc.ca/tableviewer/) are suitable for visualization of the landscape of RRIs. Here is the example:

![](http://rise.life.tsinghua.edu.cn/static/data/RRI_union_deduplicates.split_full.type_dis.human.revise.svg)

Generally we can read the txt file into pandas data frame, and use function **groupby** & **count** to count the entry for each pair of two variables. Here is the saved data frame:

```bash
[zhangqf5@loginview02 data]$ cat RRI_union_deduplicates.split_full.type_dis.human.txt
data proteinCoding lncRNA miRNA rRNA snoRNA snRNA tRNA NoncanonicalRNA others
proteinCoding 41863.0 4756.0 183.0 14128.0 115.0 721.0 66.0 3520.0 2762.0
lncRNA 9982.0 1080.0 1984.0 194.0 40.0 91.0 2.0 748.0 482.0
miRNA 35798.0 304.0 36.0 14.0 6.0 23.0 1.0 458.0 117.0
rRNA 1347.0 147.0 3.0 360.0 31.0 15.0 27.0 100.0 83.0
snoRNA 105.0 21.0 2.0 290.0 299.0 86.0 6.0 24.0 10.0
snRNA 969.0 170.0 1.0 41.0 29.0 851.0 1.0 70.0 52.0
tRNA 22.0 1.0 0.0 3.0 1.0 1.0 210.0 3.0 23.0
NoncanonicalRNA 6010.0 705.0 1076.0 69.0 104.0 69.0 1.0 1132.0 271.0
others 2850.0 384.0 328.0 74.0 12.0 43.0 5.0 262.0 400.0
```

Then we can add header to define color for each element (RNA type here):

```bash
data 1 2 3 4 5 6 7 8 9
data 202,75,78 83,169,102 205,185,111 98,180,208 129,112,182 238,130,238 255,140,0 74,113,178 169,169,169
```

Once concante header and data frame, the combined data is ready for plot:

```bash
[zhangqf5@loginview02 data]$ cat RRI_union_deduplicates.split_full.type_dis.human.txt
data 1 2 3 4 5 6 7 8 9
data 202,75,78 83,169,102 205,185,111 98,180,208 129,112,182 238,130,238 255,140,0 74,113,178 169,169,169
data proteinCoding lncRNA miRNA rRNA snoRNA snRNA tRNA NoncanonicalRNA others
proteinCoding 41863.0 4756.0 183.0 14128.0 115.0 721.0 66.0 3520.0 2762.0
lncRNA 9982.0 1080.0 1984.0 194.0 40.0 91.0 2.0 748.0 482.0
miRNA 35798.0 304.0 36.0 14.0 6.0 23.0 1.0 458.0 117.0
rRNA 1347.0 147.0 3.0 360.0 31.0 15.0 27.0 100.0 83.0
snoRNA 105.0 21.0 2.0 290.0 299.0 86.0 6.0 24.0 10.0
snRNA 969.0 170.0 1.0 41.0 29.0 851.0 1.0 70.0 52.0
tRNA 22.0 1.0 0.0 3.0 1.0 1.0 210.0 3.0 23.0
NoncanonicalRNA 6010.0 705.0 1076.0 69.0 104.0 69.0 1.0 1132.0 271.0
others 2850.0 384.0 328.0 74.0 12.0 43.0 5.0 262.0 400.0
```

We get plot like this:

![img](/assets/RRI_union_deduplicates.split_full.type_dis.human.svg)

However, there is a issue. The connection actually has no direction, thus the ribbon color from RNA1 to RNA2 must be the same as from RNA2 to RNA1. In the graph above, for example, there are two bands connect mRNA (red) and others (grey), but these two colors are different. In this case, we need to parse the data table to keep all count values apear in one side of the diagonal (upper or lower).

After revise, we get the data with all value on upper triangle:

```bash
[zhangqf5@loginview02 data]$ cat RRI_union_deduplicates.split_full.type_dis.human.revise.txt
data 1 2 3 4 5 6 7 8 9
data 202,75,78 83,169,102 205,185,111 98,180,208 129,112,182 238,130,238 255,140,0 74,113,178 169,169,169
data proteinCoding lncRNA miRNA rRNA snoRNA snRNA tRNA NoncanonicalRNA others
proteinCoding 41863.0 14738.0 35981.0 15475.0 220.0 1690.0 88.0 9530.0 5612.0
lncRNA 0 1080.0 2288.0 341.0 61.0 261.0 3.0 1453.0 866.0
miRNA 0 0 36.0 17.0 8.0 24.0 1.0 1534.0 445.0
rRNA 0 0 0 360.0 321.0 56.0 30.0 169.0 157.0
snoRNA 0 0 0 0 299.0 115.0 7.0 128.0 22.0
snRNA 0 0 0 0 0 851.0 2.0 139.0 95.0
tRNA 0 0 0 0 0 0 210.0 4.0 28.0
NoncanonicalRNA 0 0 0 0 0 0 0 1132.0 533.0
others 0 0 0 0 0 0 0 0 400.0
```

Then using these data we can connect the elements without direction as below:

![](http://rise.life.tsinghua.edu.cn/static/data/RRI_union_deduplicates.split_full.type_dis.human.revise.svg)