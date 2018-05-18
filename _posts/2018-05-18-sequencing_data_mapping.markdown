---
layout: post
category: "genomics"
title:  "Mapping sequencing data"
tags: [genomics, map]
---

### generate genome index for subsequent mapping

#### STAR index

```bash
runThreadN=12
genomeDir=/Share/home/zhangqf7/gongjing/zebrafish/data/reference/gtf/xiongtl/refseq_ensembl_homolog
genomeFastaFiles=/Share/home/zhangqf7/gongjing/zebrafish/data/reference/gtf/xiongtl/refseq_ensembl_homolog.fa

STAR --runThreadN $runThreadN \
     --runMode genomeGenerate \
     --genomeDir $genomeDir \
     --genomeFastaFiles $genomeFastaFiles
```

output files:

```bash
[zhangqf7@bnode02 xiongtl]$ ll
total 1.1G
-rw-rw----+ 1 zhangqf7 zhangqf 8.3M May 18 15:31 Log.out
drwxrwx---+ 2 zhangqf7 zhangqf 4.0K May 18 15:31 refseq_ensembl_homolog
-rw-rw----+ 1 zhangqf7 zhangqf 139M May 18 15:23 refseq_ensembl_homolog.fa
-rwxrw----+ 1 zhangqf7 zhangqf  362 May 18 15:30 star_index.sh
drwx------+ 2 zhangqf7 zhangqf 4.0K May 18 15:31 _STARtmp
```
