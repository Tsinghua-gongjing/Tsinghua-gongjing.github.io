---
layout: post
category: "linux"
title:  "Linux common used tricks"
tags: [linux]
---

- TOC
{:toc}

## Rename multiple filenames [reference](https://www.tecmint.com/rename-multiple-files-in-linux/)

```bash
# rename 's/old-name/new-name/' files

gongjing@hekekedeiMac ..ject/meme_img % ll
total 1.3M
-rw-r--r-- 1 gongjing staff 14K Jun 19 14:47 logoRBNS_A1CF.eps
-rw-r--r-- 1 gongjing staff 14K Jun 19 14:47 logoRBNS_BOLL.eps
-rw-r--r-- 1 gongjing staff 14K Jun 19 14:47 logoRBNS_CELF1.eps

gongjing@hekekedeiMac ..ject/meme_img % rename 's/logoRBNS_//' *eps

gongjing@hekekedeiMac ..ject/meme_img % ll
total 1.3M
-rw-r--r-- 1 gongjing staff 14K Jun 19 14:47 A1CF.eps
-rw-r--r-- 1 gongjing staff 14K Jun 19 14:47 BOLL.eps
-rw-r--r-- 1 gongjing staff 14K Jun 19 14:47 CELF1.eps

gongjing@hekekedeiMac ..ject/meme_img % rename 's/.eps//' *eps

gongjing@hekekedeiMac ..ject/meme_img % ll
total 1.3M
-rw-r--r-- 1 gongjing staff 14K Jun 19 14:47 A1CF
-rw-r--r-- 1 gongjing staff 14K Jun 19 14:47 BOLL
-rw-r--r-- 1 gongjing staff 14K Jun 19 14:47 CELF1
```

## Drop specific columns [linuxconfig](https://linuxconfig.org/how-to-remove-columns-from-csv-based-on-column-number-using-bash-shell)

```bash
### use cut -f and --complement

[zhangqf5@loginview02 RBPgroup_CLIPseq_hg38_anno]$ head test.txt|awk -F "\t" '{print NF}'
31
31
31
31
31
31
31
31
31
31

[zhangqf5@loginview02 RBPgroup_CLIPseq_hg38_anno]$ head test.txt|awk 'BEGIN{OFS="\t";}{if($17==10)print}'|cut -f17 --complement|awk -F "\t" '{print NF}'
30
30
30
30
30
30
30
30
```