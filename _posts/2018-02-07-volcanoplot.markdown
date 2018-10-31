---
layout: post
category: "visualization"
title:  "Volcano plot"
tags: [plot, visualization]
---

### R version [reference](http://www.gettinggeneticsdone.com/2014/05/r-volcano-plots-to-visualize-rnaseq-microarray.html)

Example data can be download from [here](https://gist.githubusercontent.com/stephenturner/806e31fce55a8b7175af/raw/9479acb809fae09aa50ea7df54a1199b3f1ffa11/results.txt) and save as `results.txt`

```bash
$ head results.txt

Gene log2FoldChange pvalue padj
DOK6 0.51 1.861e-08 0.0003053
TBX5 -2.129 5.655e-08 0.0004191
SLC32A1 0.9003 7.664e-08 0.0004191
IFITM1 -1.687 3.735e-06 0.006809
NUP93 0.3659 3.373e-06 0.006809
```

Plot valcano:

```R
# Make a basic volcano plot
with(res, plot(log2FoldChange, -log10(pvalue), pch=20, main="Volcano plot", xlim=c(-2.5,2)))

# Add colored points: red if padj<0.05, orange of log2FC>1, green if both)
with(subset(res, padj<.05 ), points(log2FoldChange, -log10(pvalue), pch=20, col="red"))
with(subset(res, abs(log2FoldChange)>1), points(log2FoldChange, -log10(pvalue), pch=20, col="orange"))
with(subset(res, padj<.05 & abs(log2FoldChange)>1), points(log2FoldChange, -log10(pvalue), pch=20, col="green"))

# Label points with the textxy function from the calibrate plot
library(calibrate)
with(subset(res, padj<.05 & abs(log2FoldChange)>1), textxy(log2FoldChange, -log10(pvalue), labs=Gene, cex=.8))
```