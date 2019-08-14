---
layout: post
category: "linux"
title:  "Job manager on cluster"
tags: [linux]
---

- TOC
{:toc}

---

## IBM对于各命令的解释

* [bsub](https://www.ibm.com/support/knowledgecenter/en/SSETD4_9.1.2/lsf_command_ref/bsub.1.html): Submits a job to LSF
* [bjobs](https://www.ibm.com/support/knowledgecenter/en/SSETD4_9.1.2/lsf_command_ref/bjobs.1.html): displays and filters information about LSF jobs
* [bqueues](https://www.ibm.com/support/knowledgecenter/en/SSETD4_9.1.3/lsf_command_ref/bqueues.1.html): displays information about queues

---

## 查看运行的任务

* `bjobs`: 列出正在运行的任务
* `-w`: 展示全部的信息，尤其是任务名称，在识别不同的样本时很有用

```bash
$ bjobs -w
JOBID   USER    STAT  QUEUE      FROM_HOST   EXEC_HOST   JOB_NAME   SUBMIT_TIME
734009  zhangqf7 RUN   Z-ZQF      loginview02 5*node524   Fastqc_raw.4 Apr 23 10:58
734092  zhangqf7 RUN   Z-ZQF      loginview02 5*node526   Fastqc_raw.11 Apr 23 10:58
734117  zhangqf7 RUN   Z-ZQF      loginview02 5*node522   Fastqc_raw.13 Apr 23 10:58
734212  zhangqf7 RUN   Z-ZQF      loginview02 5*node531   Fastqc_raw.21 Apr 23 10:58
```

---

## 提交含有脚本参数的任务

有时候需要提交的任务除了qsub可以指定的参数之外，还有本身执行的命令带有的命令行参数，比如下面例子中R脚本的参数：`-f`,`-o`，通常的做法是，用引号把执行的命令部分引起来，格式如下（可参考[这里](https://unix.stackexchange.com/questions/144518/pass-argument-to-script-then-redirect-script-as-input-to-bsub)）：

`bsub -q [queue] -J "[name]" -W 00:10 [other bsub args] "sh script.sh [script args]"`


```bash
sample=test
bsub_err=./${sample}.err
bsub_out=./${sample}.out
filein=/Share/home/zhangqf5/gongjing/data/${sample}.txt
fileout=/Share/home/zhangqf5/gongjing/rerun/${sample}
bsub -q Z-ZQF -oo $bsub_out -eo $bsub_err "Rscript test.R -f $filein -o $fileout"
```

---

## 查看集群队列使用情况

```bash
$ bqueues
QUEUE_NAME      PRIO STATUS          MAX JL/U JL/P JL/H NJOBS  PEND   RUN  SUSP
TEST             60  Open:Active       -    -    -    -     0     0     0     0
TEST 1          60  Open:Active     460    -    -    -   380   380     0     0
TEST 2          60  Open:Active     380    -    -    -   680   580   100     0
```

---

