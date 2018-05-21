---
layout: post
category: "genomics"
title:  "Using DESeq2 to do differential expression analysis"
tags: [genomics, sequencing]
---

### DESeq2

#### Calling differential expression (DE) genes based on read count among different conditions, for instance, medicine treatment versus mock. Generally we used both fold change and pvalue to filter DE genes, e.g., |log2(Fold change)| >=2 and pvalue <=0.05. Here is an simple script to run DE analysis as described [here: Template for analysis with DESeq2](https://gist.github.com/stephenturner/f60c1934405c127f09a6).

#### Prepare read count data

```bash
[zhangqf7@loginview02 RIP]$ head readcount_h2.txt
NM_212847       1922.0  3057.0
NM_212844       1552.0  245.0
NM_001004627    1901.0  22507.0
NM_212849       15338.0 2889.0
NM_205638       3326.0  13646.0
``` 

#### Load read count and condition data, run the DE analysis

```R
library("DESeq2")

countdata <- read.table("/Share/home/zhangqf7/gongjing/zebrafish/data/RIP/readcount_h2.txt", header=FALSE, row.names=1)
colnames(countdata) = c("HuR", "Control")
countdata <- as.matrix(countdata)

condition = factor(c("HuR", "Control"))
(coldata <- data.frame(row.names=colnames(countdata), condition))
dds <- DESeqDataSetFromMatrix(countData=countdata, colData=coldata, design=~condition)

dds <- DESeq(dds)

# Get differential expression results
res <- results(dds)
#table(res$padj<0.05)
## Order by adjusted p-value
res <- res[order(res$padj), ]
## Merge with normalized count data
resdata <- merge(as.data.frame(res), as.data.frame(counts(dds, normalized=TRUE)), by="row.names", sort=FALSE)
names(resdata)[1] <- "Transcript"
head(resdata)

## filter by pvalue or any criteria else
resdata = resdata[resdata$pvalue<0.05,]
## Write results
write.csv(resdata, file="/Share/home/zhangqf7/gongjing/zebrafish/data/RIP/readcount_h2.DE.csv")
```

Finally the content of result object `resdata`, and can be saved to a csv file: 

```
          Transcript   baseMean log2FoldChange    lfcSE      stat     pvalue
1          NM_212844  1139.1833       2.539801 1.121616  2.264413 0.02354874
2       NM_001004627  9612.8973      -2.147552 1.010351 -2.125550 0.03354078
3          NM_212849 11431.3248       2.497761 1.064955  2.345415 0.01900593
4       NM_001159988  1066.9555      -2.126481 1.068342 -1.990450 0.04654139
5       NM_001159983  1337.8418      -2.272092 1.083839 -2.096337 0.03605233
6 ENSDART00000172192   307.6127      -2.232198 1.097730 -2.033467 0.04200536
       padj         HuR    Control
1 0.3812277  2097.04486   181.3218
2 0.3812277  2568.60972 16657.1849
3 0.3812277 20724.53227  2138.1173
4 0.3812277   268.88655  1865.0245
5 0.3812277   291.85676  2383.8269
6 0.3812277    67.55943   547.6659
```
