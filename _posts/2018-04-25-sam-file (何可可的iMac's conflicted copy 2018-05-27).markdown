---
layout: post
category: "genomics"
title:  "Sam file related"
tags: [genomics, sam, bam]
---

# file format

1. header lines

	![img](https://image.slidesharecdn.com/epizoneformats-160420093442/95/ngs-data-formats-and-analyses-16-638.jpg?cb=1461145216)

2. meaning of each column:


	![img](http://felixfan.github.io/figure2016/SAMv1_3.png)

3. FLAG: how is the read mapped ?

   [explain-flags from broad institue](https://broadinstitute.github.io/picard/explain-flags.html)
   
   ![img](https://ppotato.files.wordpress.com/2010/08/sam_output2.png)
   
   
   
# one liner parser
 
 1. [SAM and BAM filtering oneliners (github)](https://gist.github.com/davfre/8596159) 

 
# example operation

#### Extracting reads for a single chromosome from BAM/SAM [source](https://carleshf87.wordpress.com/2013/10/28/extracting-reads-for-a-single-chromosome-from-bamsam-file-with-samtools/)

`If no regions or options: print all`

`If specific one or more regions (space separted): print restricted ` Note: **need sorted and indexed**


```bash
samtools view -h hur_MO-h6_rep2.sorted.bam NM_001002366 > test.sam

# convert bam directly
samtools view -bS HG00096.chr20.sam > HG00096.chr20.bam

# specific multiple chromosome or regions by space
samtools view -bS *bam chr1 chr2 chr3 > test.bam
```