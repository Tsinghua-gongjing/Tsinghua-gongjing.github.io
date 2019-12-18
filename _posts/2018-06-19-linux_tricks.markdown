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

---

## 获取脚本参数

参考[这里](https://stackoverflow.com/questions/3811345/how-to-pass-all-arguments-passed-to-my-bash-script-to-a-function-of-mine/3816747):

```bash
# 这里第一个参数是GPU ID，第2个及之后的参数是要训练的文件夹名称（遍历训练）
for i in "${@:2}"
do
echo "process: "$i", with GPU: "$1
cd $expr/$i
bash train.sh $1
done

# $1: 第一个参数
# "${@:2}": 第二个及之后的所有参数
# "$@": 获得所有参数
# "${@:3:4}": 第3个参数开始的4个参数，即$3,$4,$5,$6
```

---

## ImportError: /lib64/libc.so.6: version `GLIBC_2.14' not found

```bash
### https://zhuanlan.zhihu.com/p/40444240

# 检查系统含有的GLIBC版本
$ strings /lib64/libc.so.6 |grep GLIBC

GLIBC_2.2.5
GLIBC_2.2.6
GLIBC_2.3
GLIBC_2.3.2
GLIBC_2.3.3
GLIBC_2.3.4
GLIBC_2.4
GLIBC_2.5
GLIBC_2.6
GLIBC_2.7
GLIBC_2.8
GLIBC_2.9
GLIBC_2.10
GLIBC_2.11
GLIBC_2.12
GLIBC_PRIVATE

# 下载、安装
$ wget https://ftp.gnu.org/gnu/glibc/glibc-2.14.tar.gz
$ tar zxf glibc-2.14.tar.gz
$ cd glibc-2.14
$ mkdir build
$ cd build
$ ../configure --prefix=/opt/glibc-2.14
$ make -j4
$ make install


# 添加到LD_LIBRARYPATH
$ export LD_LIBRARY_PATH=/opt/glibc-2.14/lib:$LD_LIBRARY_PATH
```

---

## `ssh`免密码登录

网上教程很多，随便参考，比如[这里](https://www.linuxdashen.com/ssh-key%EF%BC%9A%E4%B8%A4%E4%B8%AA%E7%AE%80%E5%8D%95%E6%AD%A5%E9%AA%A4%E5%AE%9E%E7%8E%B0ssh%E6%97%A0%E5%AF%86%E7%A0%81%E7%99%BB%E5%BD%95)：

在自己的Linux系统上生成SSH密钥和公钥

```bash
➜  ~ ssh-keygen
Generating public/private rsa key pair.
Enter file in which to save the key (/home/gongjing/.ssh/id_rsa):
/home/gongjing/.ssh/id_rsa already exists.
Overwrite (y/n)? y
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/gongjing/.ssh/id_rsa.
Your public key has been saved in /home/gongjing/.ssh/id_rsa.pub.
The key fingerprint is:
SHA256:RWPGrhUHIbbp9DvzJX2cUOrzyFyibEdd8qU/7Mi5KUU gongjing@omnisky
The key's randomart image is:
+---[RSA 2048]----+
|        o.Bo     |
|       . Bo..    |
|        +..o   . |
|       o oo  E+ o|
|        So. .o.+o|
|        .  ..+ooo|
|          + ++++o|
|          .*=+X+.|
|          .o+Xoo.|
+----[SHA256]-----+
```

将SSH公钥上传到Linux服务器(写在对方的`~/.ssh/authorized_keys`中)

```bash
➜  ~ ssh-copy-id gongjing@10.10.91.12
/usr/bin/ssh-copy-id: INFO: Source of key(s) to be installed: "/home/gongjing/.ssh/id_rsa.pub"
/usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s), to filter out any that are already installed
/usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed -- if you are prompted now it is to install the new keys
gongjing@10.10.91.12's password:

Number of key(s) added: 1

Now try logging into the machine, with:   "ssh 'gongjing@10.10.91.12'"
and check to make sure that only the key(s) you wanted were added.
```

如果`id_rsa`之前是其他用户（比如root）创建的，有可能会出现权限错误：

```bash
➜  ~ ssh gongjing@10.10.91.12
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@         WARNING: UNPROTECTED PRIVATE KEY FILE!          @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Permissions 0755 for '/home/gongjing/.ssh/id_rsa' are too open.
It is required that your private key files are NOT accessible by others.
This private key will be ignored.
Load key "/home/gongjing/.ssh/id_rsa": bad permissions
gongjing@10.10.91.12's password:
```

此时可以更改权限，使得文件`id_rsa`只有自己可读写(参考[这里](https://stackoverflow.com/questions/9270734/ssh-permissions-are-too-open-error))：

```bash
chmod 600 ~/.ssh/id_rsa
```

---

### rsync同步文件

使用方法可参考[这里](https://www.digitalocean.com/community/tutorials/how-to-use-rsync-to-sync-local-and-remote-directories-on-a-vps)：

* 参数`-a`：迭代同步，子目录、文件链接等都会同步
* 参数`-P`：显示同步进度，速率、已完成、剩余等

```bash
# 同步文件夹
rsync -aP download_20191204 gongjing@10.10.91.12:/home/gongjing/

# 同步文件
rsync -aP ./Untitled.ipynb gongjing@10.10.91.12:/home/gongjing/scripts
```