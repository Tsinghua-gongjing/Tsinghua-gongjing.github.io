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
|mouse (mm9)|[NCBIM37.genome.fa](/Share/home/zhangqf/database/GenomeAnnotation/genome/NCBIM37.genome.fa)|[GRCm37.gtf](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/GRCm37.gtf)||[mm9.genomeCoor.bed](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/mm9.genomeCoor.bed), [mm9.transCoor.bed](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/mm9.transCoor.bed)|[mm9_transcriptome.fa](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/mm9_transcriptome.fa)|
|mouse (mm10)|[GRCm38.p5.genome.fa](/Share/home/zhangqf/database/GenomeAnnotation/genome/GRCm38.p5.genome.fa)|[GRCm38.gtf](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/GRCm38.gtf)||[mm10.genomeCoor.bed](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/mm10.genomeCoor.bed), [mm10.transCoor.bed](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/mm10.transCoor.bed)|[mm10_transcriptome.fa](/Share/home/zhangqf/database/GenomeAnnotation/Gencode/mm10_transcriptome.fa)|

Note: 

Genome sequences are assume consistent between different sources.

------------------------------------------------------------------------------------------------


### NCBI download

|Species|Genome|GTF|GFF|BED|Transcriptome|
|---|---|---|---|---|---|
|human (hg19)|[]()|[]()|[GRCh37.gff3](/Share/home/zhangqf/database/GenomeAnnotation/NCBI/GRCh37.gff3)|[hg19.genomeCoor.bed](/Share/home/zhangqf/database/GenomeAnnotation/NCBI/hg19.genomeCoor.bed), [hg19.transCoor.bed](/Share/home/zhangqf/database/GenomeAnnotation/NCBI/hg19.transCoor.bed)|[hg19_transcriptome.fa](/Share/home/zhangqf/database/GenomeAnnotation/NCBI/hg19_transcriptome.fa)|
|human (hg38)|[]()|[]()|[GRCh38.gff3](/Share/home/zhangqf/database/GenomeAnnotation/NCBI/GRCh38.gff3)|[hg38.genomeCoor.bed](/Share/home/zhangqf/database/GenomeAnnotation/NCBI/hg38.genomeCoor.bed), [hg38.transCoor.bed](/Share/home/zhangqf/database/GenomeAnnotation/NCBI/hg38.transCoor.bed)|[hg38_transcriptome.fa](/Share/home/zhangqf/database/GenomeAnnotation/NCBI/hg38_transcriptome.fa)|
|mouse (mm10)|[]()|[]()|[GRCm38.gff3](/Share/home/zhangqf/database/GenomeAnnotation/NCBI/GRCm38.gff3)|[mm10.genomeCoor.bed](/Share/home/zhangqf/database/GenomeAnnotation/NCBI/mm10.genomeCoor.bed), [mm10.transCoor.bed](/Share/home/zhangqf/database/GenomeAnnotation/NCBI/mm10.transCoor.bed)|[mm10_transcriptome.fa](/Share/home/zhangqf/database/GenomeAnnotation/NCBI/mm10_transcriptome.fa)|


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
