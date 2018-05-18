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

I wrote a script to call tableviewer:

```python
import subprocess, os
import sys

def tableview(txt=None, parse_table=None, make_conf=None, conf_dir=None, circos_conf=None, save_dir=None, parsed_conf=None):
        if parse_table is None:
                parse_table = '/Share/home/zhangqf5/gongjing/software/circos-tools-0.22/tools/tableviewer/bin/parse-table'
        if make_conf is None:
                make_conf = '/Share/home/zhangqf5/gongjing/software/circos-tools-0.22/tools/tableviewer/bin/make-conf'
        if conf_dir is None:
                conf_dir = '/Share/home/zhangqf5/gongjing/software/circos-tools-0.22/tools/tableviewer2/data'
        if circos_conf is None:
                circos_conf = '/Share/home/zhangqf5/gongjing/software/circos-tools-0.22/tools/tableviewer2/etc/circos.conf'
        if save_dir is None:
                save_dir = os.path.dirname(txt)
        if parsed_conf is None:
                #parsed_conf = '/Share/home/zhangqf5/gongjing/software/circos-tools-0.22/tools/tableviewer/samples/parse-table-02a.conf'
                parsed_conf = '/Share/home/zhangqf5/gongjing/software/circos-tools-0.22/tools/tableviewer2/etc/parse-table.conf'
        print "[tableview start] file: %s"%(txt)
        subprocess.call(["cat {txt} | {parse_table} -conf {parsed_conf} -segment_order=ascii,size_desc -placement_order=row,col -interpolate_type count -col_order_row -use_col_order_row -col_color_row -use_col_color_row -ribbon_layer_order=size_asc | {make_conf} -dir {conf_dir}".format(txt=txt, parse_table=parse_table, parsed_conf=parsed_conf, make_conf=make_conf, conf_dir=conf_dir)],shell=True)
        #subprocess.call(["cat {txt} | {parse_table} -conf {parsed_conf} | {make_conf} -dir {conf_dir}".format(txt=txt, parse_table=parse_table, parsed_conf=parsed_conf, make_conf=make_conf, conf_dir=conf_dir)],shell=True)
        file_png = txt.split('/')[-1].replace('txt', 'png')
        subprocess.call(["circos -conf {circos_conf} -outputdir {save_dir} -outputfile {file_png} -param random_string=zgvickusamp| grep created".format(circos_conf=circos_conf, save_dir=save_dir, file_png=file_png)],shell=True)
        print "[tableview end] file: %s"%(txt)

def main():
        #tableview(txt='/Share/home/zhangqf5/gongjing/software/circos-tools-0.22/tools/tableviewer/samples/RRI_union_deduplicates.split_full.type_dis.txt')
        if len(sys.argv) == 1:
            txt = '/Share/home/zhangqf5/gongjing/software/circos-tools-0.22/tools/tableviewer2/samples/RRI_union_deduplicates.split_full.type_dis.txt'
        else:
            txt = sys.argv[1]
        tableview(txt)

if __name__ == '__main__':
        main()
```

Run the script with previous data:

```bash
[zhangqf5@loginview02 tableviewer2]$ pwd
/Share/home/zhangqf5/gongjing/software/circos-tools-0.22/tools/tableviewer2
[zhangqf5@loginview02 tableviewer2]$ ll
total 40K
drwxr-x--- 2 zhangqf5 zhangqf 4.0K Jul  6  2017 batch
drwxr-x--- 2 zhangqf5 zhangqf 4.0K Aug  2  2017 bin
drwxr-x--- 2 zhangqf5 zhangqf 4.0K May 19 02:21 data
drwxr-x--- 2 zhangqf5 zhangqf 4.0K May 19 02:21 etc
drwxr-x--- 2 zhangqf5 zhangqf 4.0K Jul  6  2017 img
drwxr-x--- 2 zhangqf5 zhangqf 4.0K Jul  6  2017 lib
-rwxr----- 1 zhangqf5 zhangqf 2.4K Jul  6  2017 makeimage.py
drwxr-x--- 2 zhangqf5 zhangqf 4.0K Jul  6  2017 results
drwxr-x--- 2 zhangqf5 zhangqf 4.0K Jul  6  2017 samples
drwxr-x--- 2 zhangqf5 zhangqf 4.0K Jul  6  2017 uploads

[zhangqf5@loginview02 tableviewer2]$ python makeimage.py /Share/home/zhangqf5/gongjing/DNA-RNA-Protein-interaction-correlation-12-18/data/RRI_union_deduplicates.split_full.type_dis.human.txt
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