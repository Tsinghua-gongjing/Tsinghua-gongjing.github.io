---
layout: post
category: "python"
title:  "Comparison between Python and R"
tags: [python, R]
---

## [NumPy for R (and S-Plus) users](http://mathesaurus.sourceforge.net/r-numpy.html)

This resource compare the basic usage in both languages, which is pretty direct.


## R usage

### 字符串替换

函数：`gsub(pattern, replacement, x)`，例子[如下](https://stackoverflow.com/questions/11936339/replace-specific-characters-within-strings):

```R
group <- c("12357e", "12575e", "197e18", "e18947")
group
[1] "12357e" "12575e" "197e18" "e18947"

gsub("e", "", group)
[1] "12357" "12575" "19718" "18947"
```

### R脚本获取命令行参数

函数：`commandArgs`，例子[如下](https://stackoverflow.com/questions/14167178/passing-command-line-arguments-to-r-cmd-batch)：

```R
args <- commandArgs(trailingOnly = TRUE)
print(args[1]) # args[0] -> script anme
```

### 判断字符串收尾是否含有特定字符

函数：`startsWith`，`endsWith`，例子[如下](https://stackoverflow.com/questions/31467732/does-r-have-function-startswith-or-endswith-like-python)：

```R
> startsWith("what", "wha")
[1] TRUE
> startsWith("what", "ha")
[1] FALSE
```

### 示例脚本：含有spike-ins的RNAseq差异分析

```R
args<-commandArgs(TRUE)

library(RUVSeq)

# f='Control_MO_H4_H6.txt'
f=args[1]
zfGenes = read.table(f, header=T, row.names=1, sep='\t')

filter <- apply(zfGenes, 1, function(x) length(x[x>5])>=2)
filtered <- zfGenes[filter,]
genes <- rownames(filtered)[grep("^N", rownames(filtered))]
spikes <- rownames(filtered)[grep("^ERCC", rownames(filtered))]

condition1=args[2]
condition2=args[3]

x <- as.factor(rep(c(condition1, condition2), each=2))
print(x)
set <- newSeqExpressionSet(as.matrix(filtered), phenoData = data.frame(x, row.names=colnames(filtered)))
head(counts(set))

set1 <- RUVg(set, spikes, k=1)

library(DESeq2)
dds <- DESeqDataSetFromMatrix(countData = counts(set1), colData = pData(set1), design = ~ W_1 + x) # use spike-in adjust
# dds <- DESeqDataSetFromMatrix(countData = counts(set), colData = pData(set), design = ~x) # use not adjust
dds <- DESeq(dds)
res <- results(dds)

res <- res[order(res$padj), ]
## Merge with normalized count data
resdata <- merge(as.data.frame(res), as.data.frame(counts(dds, normalized=TRUE)), by="row.names", sort=FALSE)
names(resdata)[1] <- "Transcript"
head(resdata)
resdata = resdata[resdata$pvalue>=0,]
## Write results
savefn=args[4]
write.csv(resdata, file=savefn)
```
