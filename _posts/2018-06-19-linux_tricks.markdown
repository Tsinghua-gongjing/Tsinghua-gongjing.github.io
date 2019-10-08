---
layout: post
category: "linux"
title:  "Linux common used tricks"
tags: [linux]
---

- TOC
{:toc}

---

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

有的时候`rename`命令不正常，可能是因为使用的命令来源不同，就像[这里](https://askubuntu.com/questions/1024960/rename-not-working)讨论的：

首先要查看环境的`rename`是来源，然后写对应的命令，不能弄错了：

```bash
$ rename --version
/usr/bin/rename using File::Rename version 0.20
$ rename 's/\.jpeg$/.jpg/' *

$ rename --version
rename from util-linux 2.30.2
$ rename .jpeg .jpg *

# 集群上使用的是util-linux这个版本
# 命令：rename pattern replace files
$ rename --version
rename (util-linux-ng 2.17.2)
```

---

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

---

## Mac OS general compress/decompress command line tool
```bash
# https://theunarchiver.com/command-line
$ wget https://cdn.theunarchiver.com/downloads/unarMac.zip

$ ll
-rwxr-xr-x   1 gongjing  staff   1.8M May 19  2016 lsar
-rwxr-xr-x   1 gongjing  staff   1.8M May 19  2016 unar

$ unar
unar v1.10.1 (May 19 2016), a tool for extracting the contents of archive files.
Usage: unar [options] archive [files ...]

Available options:
-output-directory (-o) <string>         The directory to write the contents of the archive to. Defaults to the current directory. If set to a single dash (-), no files will be
                                        created, and all data will be output to stdout.
-force-overwrite (-f)                   Always overwrite files when a file to be unpacked already exists on disk. By default, the program asks the user if possible, otherwise skips
                                        the file.
-force-rename (-r)                      Always rename files when a file to be unpacked already exists on disk.
-force-skip (-s)                        Always skip files when a file to be unpacked already exists on disk.
-force-directory (-d)                   Always create a containing directory for the contents of the unpacked archive. By default, a directory is created if there is more than one
                                        top-level file or folder.
-no-directory (-D)                      Never create a containing directory for the contents of the unpacked archive.
-password (-p) <string>                 The password to use for decrypting protected archives.
-encoding (-e) <encoding name>          The encoding to use for filenames in the archive, when it is not known. If not specified, the program attempts to auto-detect the encoding
                                        used. Use "help" or "list" as the argument to give a listing of all supported encodings.
-password-encoding (-E) <name>          The encoding to use for the password for the archive, when it is not known. If not specified, then either the encoding given by the -encoding
                                        option or the auto-detected encoding is used.
-indexes (-i)                           Instead of specifying the files to unpack as filenames or wildcard patterns, specify them as indexes, as output by lsar.
-no-recursion (-nr)                     Do not attempt to extract archives contained in other archives. For instance, when unpacking a .tar.gz file, only unpack the .gz file and not
                                        its contents.
-copy-time (-t)                         Copy the file modification time from the archive file to the containing directory, if one is created.
-no-quarantine (-nq)                    Do not copy Finder quarantine metadata from the archive to the extracted files.
-forks (-k) <fork|visible|hidden|skip>  How to handle Mac OS resource forks. "fork" creates regular resource forks, "visible" creates AppleDouble files with the extension ".rsrc",
                                        "hidden" creates AppleDouble files with the prefix "._", and "skip" discards all resource forks. Defaults to "fork".
-quiet (-q)                             Run in quiet mode.
-version (-v)                           Print version and exit.
-help (-h)                              Display this information.
```

---

## Iterate over two arrays [stackoverflow](https://stackoverflow.com/questions/17403498/iterate-over-two-arrays-simultaneously-in-bash)

```bash
#!/bin/bash

array=( "Vietnam" "Germany" "Argentina" )
array2=( "Asia" "Europe" "America" )

for ((i=0;i<${#array[@]};++i)); do
    printf "%s is in %s\n" "${array[i]}" "${array2[i]}"
done
```

---

## GCC version

Run binary command which depends on `GLIBC_2.14` and `GLIBC_2.17`:

```bash
[zhangqf5@loginview02 bin]$ ./clan_annotate
./clan_annotate: /lib64/libc.so.6: version `GLIBC_2.14' not found (required by ./clan_annotate)
./clan_annotate: /lib64/libc.so.6: version `GLIBC_2.17' not found (required by ./clan_annotate)
```

Check GCC version:
 
```bash
[zhangqf5@loginview02 bin]$ ldd --version
ldd (GNU libc) 2.12
Copyright (C) 2010 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
Written by Roland McGrath and Ulrich Drepper.
```

---

## Print every two lines as one line

```bash
# print lines of files matched with specific pattern
[zhangqf7@loginview02 bwa]$ wl /Share/home/zhangqf7/jinsong_zhang/zebrafish/data/iclip/20181224/Rawdata/shi-zi-*/bwa/CTK_Procedure{1,2,3,4}/CITS/iCLIP.tag*p05.bed|awk '{print $1}'
15450
11876
26860
19994
69867
174233
13396
10204
24288
18042
65161
161563
2059
1369
1983
1569
6291
13434
2329
1535
2171
1758
6762
14614
666808


[zhangqf7@loginview02 bwa]$ wl /Share/home/zhangqf7/jinsong_zhang/zebrafish/data/iclip/20181224/Rawdata/shi-zi-*/bwa/CTK_Procedure{1,2,3,4}/CITS/iCLIP.tag*p05.bed|awk '{print $1}'|xargs -n4 -d'\n'
15450 11876 26860 19994
69867 174233 13396 10204
24288 18042 65161 161563
2059 1369 1983 1569
6291 13434 2329 1535
2171 1758 6762 14614
666808
```

---

## Update file soft link

Use option `-sfn` of `ln` command as discussed [here](https://serverfault.com/questions/389997/how-to-override-update-a-symlink):

```bash
ln -sfn {path/to/file-name} {link-name}
```

---

## install software instead of default dir

```bash
./configure --prefix=/somewhere/else/than/usr/local
make
make install
```

---

## 脚本中获得当前路径及文件夹名称

```bash
# get full path
work_space=$(pwd)

# get dir name instead of full path
work_dir_name=${PWD##*/}
```