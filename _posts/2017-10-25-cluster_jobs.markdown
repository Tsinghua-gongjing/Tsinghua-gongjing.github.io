---
layout: post
category: "linux"
title:  "Job manager on cluster"
tags: [linux]
---

目录

- TOC
{:toc}

---

### IBM对于各命令的解释

* [bsub](https://www.ibm.com/support/knowledgecenter/en/SSETD4_9.1.2/lsf_command_ref/bsub.1.html): Submits a job to LSF
* [bjobs](https://www.ibm.com/support/knowledgecenter/en/SSETD4_9.1.2/lsf_command_ref/bjobs.1.html): displays and filters information about LSF jobs
* [bqueues](https://www.ibm.com/support/knowledgecenter/en/SSETD4_9.1.3/lsf_command_ref/bqueues.1.html): displays information about queues

---

### 查看运行的任务

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

### 提交含有脚本参数的任务

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

### 查看集群队列使用情况

```bash
$ bqueues
QUEUE_NAME      PRIO STATUS          MAX JL/U JL/P JL/H NJOBS  PEND   RUN  SUSP
TEST             60  Open:Active       -    -    -    -     0     0     0     0
TEST 1          60  Open:Active     460    -    -    -   380   380     0     0
TEST 2          60  Open:Active     380    -    -    -   680   580   100     0
```

---

### 杀死所有pending的任务

参考[这里](https://unix.stackexchange.com/questions/315839/lsf-bkill-all-pend-jobs-without-killing-run-jobs)：

```bash
bjobs -w | grep 'PEND' | awk '{print $1}' | xargs bkill
```

---

### 提交任务在某个任务完成之后再执行

```bash
bsub -q Z-ZQF -eo run.err -oo run.out -w "done(323228)" bash sampling.sh
```

![bsub_waiting_jobs_done.png](https://i.loli.net/2020/03/19/nCKdDpazVF4kAeL.png)

---

### 提交任务在某个任务完成之后再执行(任务名称)

```bash
# map
for seed in 1234 #1234 40 9988 17181790 81910 625178 1 7829999 9029102 918029109
do
	for i in 10 20 40 #10 20 30 40 50 60 70 
	do
		# mkdir -p $sampling_dir/${sample}_${i}_${seed}/1.mapGenome
		# icSHAPE-pipe mapGenome -i $sampling_dir/${sample}_${i}_${seed}/0.rawData/nai.trimmed.fastq -o $sampling_dir/${sample}_${i}_${seed}/1.mapGenome/nai -x $star_index -p 16 --maxMMap 10
		# sleep 10

		mkdir -p $sampling_dir/${sample}_${i}_${seed}/2.tab
		bsub -q Z-ZQF -J "${sample}.${seed}.${i}.totab" -n 5 -eo $sampling_dir/${sample}_${i}_${seed}/2.tab/sam2tab.err -oo $sampling_dir/${sample}_${i}_${seed}/2.tab/sam2tab.out "icSHAPE-pipe sam2tab -in $sampling_dir/${sample}_${i}_${seed}/1.mapGenome/nai.sorted.bam -out $sampling_dir/${sample}_${i}_${seed}/2.tab/nai.tab"

		mkdir -p $sampling_dir/${sample}_${i}_${seed}/3.shape
		bsub -q Z-ZQF -n 5 -w "done(${sample}.${seed}.${i}.totab)" -J "${sample}.${seed}.${i}.tabtoscore" -eo $sampling_dir/${sample}_${i}_${seed}/3.shape/score.err -oo $sampling_dir/${sample}_${i}_${seed}/3.shape/score.out \
		    "icSHAPE-pipe calcSHAPENoCont \
		        -N $sampling_dir/${sample}_${i}_${seed}/2.tab/nai.tab \
		        -size $star_index/chrNameLength.txt \
		        -out $sampling_dir/${sample}_${i}_${seed}/3.shape/score.tab \
		        -genome $ref_fa \
		        -bases A,C \
		        -non-sliding"


		for c in 200 #0 50 100 150 200 250
		do
			for T in 2 #0 1 2 3
			do
				echo $c,$T
				bsub -q Z-ZQF -w "done(${sample}.${seed}.${i}.totab) && done(${sample}.${seed}.${i}.tabtoscore)" -J "${sample}.${seed}.${i}.scoretoshape" -eo $sampling_dir/${sample}_${i}_${seed}/3.shape/genSHAPEToTransSHAPE.c${c}T${T}.err -oo $sampling_dir/${sample}_${i}_${seed}/3.shape/genSHAPEToTransSHAPE.c${c}T${T}.out \
					"icSHAPE-pipe genSHAPEToTransSHAPE -s $star_index/chrNameLength.txt -i $sampling_dir/${sample}_${i}_${seed}/3.shape/score.tab \
					-o $sampling_dir/${sample}_${i}_${seed}/3.shape/shape.c${c}T${T}M0m0.out -c $c -T $T -M 0 -m 0"
			done
		done

	done
done

```

---

### 指定需要多少核数

```bash
# 指定所需最大最小核数
bsub -n min_proc[,max_proc]
```

---

### 指定在什么时间执行

```bash
bsub -b [[year:][month:]day:]hour:minute
```

---

### 提交到所在队列的特定节点

比如有时候一个队列中某些节点内存很大，而自己的程序需要很大的内存，此时需要指定节点：

```bash
# bsub -m “host_name”
bsub -m ‘‘node1 node3’’
```

---

### 参考

* [LSF作业调度系统的使用](https://scc.ustc.edu.cn/zlsc/pxjz/201408/W020140804352832330063.pdf)
