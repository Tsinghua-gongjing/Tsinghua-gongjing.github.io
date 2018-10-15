---
layout: post
category: "genomics"
title:  "Use IGV to load .bed track file"
tags: [genomics, plot]
---

- TOC
{:toc}

# 使用IGV加载bed文件

## command line: igvtools

```bash
% ~/Downloads/IGVTools/igvtools help

Program: igvtools. IGV Version 2.3.98 (141)07/25/2017 12:12 AM

Usage: igvtools [command] [options] [input file/dir] [other arguments]

Command: version print the version number
	 sort    sort an alignment file by start position.
	 index   index an alignment file
	 toTDF    convert an input file (cn, gct, wig) to tiled data format (tdf)
	 count   compute coverage density for an alignment file
	 formatexp  center, scale, and log2 normalize an expression file
	 gui      Start the gui
	 help <command>     display this help message, or help on a specific command
	 See http://www.broadinstitute.org/software/igv/igvtools_commandline for more detailed help

Done
```

## sort and index

在加载之前，先用igvtools建立索引文件，尤其是对于比较大的bed文件，这样能节省加载时所耗用的内存，否则在加载多个track时容易出现加载失败。

在建立inde之前，需要sort bed文件，有两种方式：

```
sort -k1,1 -k2,2n  in.bed > out.sort.bed (bedtools 推荐)
sort -k1,1 -k2,2n -o in.bed in.bed  # 直接sort原文件并保存
igvtools sort in.bed out.sort.bed
igvtools index sort.bed
```


## load into IGV

在直接load进去的时候，出现报错：

[![IGV_error.jpg](https://i.loli.net/2018/02/07/5a7a9e7aef5b8.jpg)](https://i.loli.net/2018/02/07/5a7a9e7aef5b8.jpg)

原因是：所建立的index文件的路径含有中文字符，而这貌似是不支持的，在google的igv-help的群组里面也有[类似的问题](https://groups.google.com/forum/#!searchin/igv-help/java.lang.reflect.InvocationTargetException%7Csort:date/igv-help/SYtAcKTNLAk/a4ovqOBiAwAJ)如下：

[![IGV_error_solution.jpg](https://i.loli.net/2018/02/07/5a7a9f0cd0e7c.jpg)](https://i.loli.net/2018/02/07/5a7a9f0cd0e7c.jpg)


