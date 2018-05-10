---
layout: post
category: "genomics"
title:  "Frequently used genome related data"
tags: [genomics]
---

## public data

------------------------------------------------------------------------------------------------


### Gencode download

|Species|Genome|GTF|GFF|BED|Transcriptome|
|---|---|---|---|---|---|
|[human (hg19)](https://www.gencodegenes.org/releases/)|[GRCh37.p13.genome.fa](/Share/home/zhangqf/database/GenomeAnnotation/genome/GRCh37.p13.genome.fa)|[GRCh37.gtf](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/GRCh37.gtf)||[hg19.genomeCoor.bed](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/hg19.genomeCoor.bed), [hg19.transCoor.bed](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/hg19.transCoor.bed)|[hg19_transcriptome.fa](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/hg19_transcriptome.fa)|
|[human (hg38)](https://www.gencodegenes.org/releases/)|[GRCh38.p10.genome.fa](/Share/home/zhangqf/database/GenomeAnnotation/genome/GRCh38.p10.genome.fa)|[GRCh38.gtf](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/GRCh38.gtf)||[hg38.genomeCoor.bed](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/hg38.genomeCoor.bed), [hg38.transCoor.bed](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/hg38.transCoor.bed)|[hg38_transcriptome.fa](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/hg38_transcriptome.fa)|
|mouse (mm10)|[]()|---|---|---|---|
|mouse (mm9)|---|---|---|---|---|

------------------------------------------------------------------------------------------------


### NCBI download

|Species|Genome|GTF|GFF|BED|Transcriptome|
|---|---|---|---|---|---|


------------------------------------------------------------------------------------------------


### UCSC download

|Species|Genome|GTF|GFF|BED|Transcriptome|
|---|---|---|---|---|---|
|[human (hg19)](http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/)|[chromFa.tar.gz](http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz)||||[refMrna.fa.gz](http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/refMrna.fa.gz)|
|human (hg38)|---|---|---|---|---|
|mouse (mm10)|---|---|---|---|---|
|mouse (mm9)|---|---|---|---|---|
|zebrafish (z10)|---|---|---|---|---|

Note:

|file|description|
|---|---|
|chromFa.tar.gz|The assembly sequence in one file per chromosome. Repeats from RepeatMasker and Tandem Repeats Finder (with period of 12 or less) are shown in lower case; non-repeating sequence is shown in upper case.|
|refMrna.fa.gz|RefSeq mRNA from the same species as the genome.This sequence data is updated once a week via automatic GenBank updates.|


------------------------------------------------------------------------------------------------

Reference:

* [GenomeResource from Panpan](http://olddriver.website/GenomeResource/)
