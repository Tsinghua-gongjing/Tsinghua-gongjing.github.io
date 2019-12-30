---
layout: post
category: "python"
title:  "Comparison between Python and R"
tags: [python, R]
---

- TOC
{:toc}

---

## [NumPy for R (and S-Plus) users](http://mathesaurus.sourceforge.net/r-numpy.html)

This resource compare the basic usage in both languages, which is pretty direct.

---

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

---

### R脚本获取命令行参数

函数：`commandArgs`，例子[如下](https://stackoverflow.com/questions/14167178/passing-command-line-arguments-to-r-cmd-batch)：

```R
args <- commandArgs(trailingOnly = TRUE)
print(args[1]) # args[0] -> script anme
```

---

### 判断字符串收尾是否含有特定字符

函数：`startsWith`，`endsWith`，例子[如下](https://stackoverflow.com/questions/31467732/does-r-have-function-startswith-or-endswith-like-python)：

```R
> startsWith("what", "wha")
[1] TRUE
> startsWith("what", "ha")
[1] FALSE
```

---

### 读取文件时列名称保持原有特殊字符

函数：`read.table`的参数`check.names`默认对于特殊字符是要进行转换的，保持的例子[如下](https://stackoverflow.com/questions/10441437/why-am-i-getting-x-in-my-column-names-when-reading-a-data-frame)：

```R
read.csv(file, sep=",", header=T, check.names = FALSE)
```

---

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

---

### 加载`.Rdata`数据

```R
> my = load('/Share2/home/zhangqf5/gongjing/rare_cell/tools/CellSIUS/Codes_Data/input/Supplementary_File_8_sce_raw.RData')

# 查看加载的变量
> my
[1] "sce"

# 不是list形式调用
> my$sce
Error in my$sce : $ operator is invalid for atomic vectors

# 通过ls命令查看环境定义的变量
# 核对一下刚加载的是哪个
> ls()
[1] "code_dir"     "input_dir"    "my"           "out_data_dir" "plotdir"
[6] "sce"

# 查看加载的变量具有哪些属性可以使用
> attributes(sce)
$.__classVersion__
            R       Biobase          eSet ExpressionSet        SCESet
      "3.4.1"      "2.36.2"       "1.3.0"       "1.0.0"       "1.1.9"

$logExprsOffset
[1] 1

$lowerDetectionLimit
[1] 0

$cellPairwiseDistances
dist(0)

$featurePairwiseDistances
dist(0)

$reducedDimension
<0 x 0 matrix>

$bootstraps
numeric(0)

$sc3
list()

$featureControlInfo
An object of class 'AnnotatedDataFrame'
  rowNames: 1
  varLabels: name
  varMetadata: labelDescription
```

因为这个是biology的数据，所以有一些通用的属性，以及取值的方法：

```R
# 使用exprs获取count数据
# 这也是我们一般使用的expression matrix数据
# 行：gene，列：cell id
> countdata<-exprs(sce)
> dim(countdata)
[1] 23848 11680
> head(countdata,n=2)

                JRK_DA234_C2782 JRK_DA234_C2783 JRK_DA234_C2784 JRK_DA234_C2785
ENSG00000238009               0               0       0.9093836               0
ENSG00000239945               0               0       0.9093836               0


# fData: 获取feature data，这里是基因id
# 这个matrix相当于是对基因的注释，各种注释信息放进来
> fdata<-fData(sce)
> dim(fdata)
[1] 23848    15 
> head(fdata, head=2)
                        gene_id chr        symbol   gene_biotype   mean_exprs
ENSG00000238009 ENSG00000238009   1  RP11-34P13.7        lincRNA 0.0046305668
ENSG00000239945 ENSG00000239945   1  RP11-34P13.8        lincRNA 0.0004008153
ENSG00000279457 ENSG00000279457   1    FO538757.2 protein_coding 0.1828388937
ENSG00000228463 ENSG00000228463   1    AP006222.2        lincRNA 0.2726365525
ENSG00000236601 ENSG00000236601   1  RP4-669L17.2        lincRNA 0.0006401731
ENSG00000237094 ENSG00000237094   1 RP4-669L17.10        lincRNA 0.0098135864
                exprs_rank n_cells_exprs total_feature_exprs pct_total_exprs
ENSG00000238009       7274            59           54.085020    7.842510e-05
ENSG00000239945       2200             6            4.681523    6.788366e-06
ENSG00000279457      16875          2019         2135.558279    3.096632e-03
ENSG00000228463      18491          2948         3184.394934    4.617480e-03
ENSG00000236601       3144             9            7.477221    1.084222e-05
ENSG00000237094       8776           120          114.622690    1.662068e-04
                pct_dropout total_feature_counts log10_total_feature_counts
ENSG00000238009    99.49486                   61                   1.792392
ENSG00000239945    99.94863                    6                   0.845098
ENSG00000279457    82.71404                 2307                   3.363236
ENSG00000228463    74.76027                 3724                   3.571126
ENSG00000236601    99.92295                    9                   1.000000
ENSG00000237094    98.97260                  121                   2.086360
                pct_total_counts is_feature_control_MT is_feature_control
ENSG00000238009     2.977997e-05                 FALSE              FALSE
ENSG00000239945     2.929177e-06                 FALSE              FALSE
ENSG00000279457     1.126269e-03                 FALSE              FALSE
ENSG00000228463     1.818042e-03                 FALSE              FALSE
ENSG00000236601     4.393765e-06                 FALSE              FALSE
ENSG00000237094     5.907174e-05                 FALSE              FALSE

> colnames(fdata)
 [1] "gene_id"                    "chr"
 [3] "symbol"                     "gene_biotype"
 [5] "mean_exprs"                 "exprs_rank"
 [7] "n_cells_exprs"              "total_feature_exprs"
 [9] "pct_total_exprs"            "pct_dropout"
[11] "total_feature_counts"       "log10_total_feature_counts"
[13] "pct_total_counts"           "is_feature_control_MT"
[15] "is_feature_control"


# pData: 获取phenotype data，这里是cell id
# 这个matrix相当于是对细胞的注释，各种注释信息放进来
> pdata<-pData(sce)
> dim(pdata)
[1] 11680    38
> head(pdata, head=2)
                       cell_idx        Batch cell_line cell_cycle_phase    G1
IMR90_HCT116_C1 IMR90_HCT116_C1 IMR90_HCT116    HCT116               G1 1.000
IMR90_HCT116_C3 IMR90_HCT116_C3 IMR90_HCT116    HCT116               G1 0.972
IMR90_HCT116_C4 IMR90_HCT116_C4 IMR90_HCT116     IMR90               G1 0.999

> colnames(pdata)
 [1] "cell_idx"
 [2] "Batch"
 [3] "cell_line"
 [4] "cell_cycle_phase"
 [5] "G1"
 [6] "S"
 [7] "G2M"
 [8] "total_counts"
 [9] "log10_total_counts"
[10] "filter_on_total_counts"
[11] "total_features"
[12] "log10_total_features"
[13] "filter_on_total_features"
[14] "pct_dropout"
[15] "exprs_feature_controls_MT"
[16] "pct_exprs_feature_controls_MT"
[17] "filter_on_pct_exprs_feature_controls_MT"
[18] "counts_feature_controls_MT"
[19] "pct_counts_feature_controls_MT"
[20] "filter_on_pct_counts_feature_controls_MT"
[21] "n_detected_feature_controls_MT"
[22] "n_detected_feature_controls"
[23] "counts_feature_controls"
[24] "pct_counts_feature_controls"
[25] "filter_on_pct_counts_feature_controls"
[26] "pct_counts_top_50_features"
[27] "pct_counts_top_100_features"
[28] "pct_counts_top_200_features"
[29] "pct_counts_top_500_features"
[30] "pct_counts_top_50_endogenous_features"
[31] "pct_counts_top_100_endogenous_features"
[32] "pct_counts_top_200_endogenous_features"
[33] "pct_counts_top_500_endogenous_features"
[34] "counts_endogenous_features"
[35] "log10_counts_feature_controls_MT"
[36] "log10_counts_feature_controls"
[37] "log10_counts_endogenous_features"
[38] "is_cell_control"
```

---

### 查看模块和环境版本

```R
# 查看单个加载的模块的版本
> packageVersion('scater')
[1] ‘1.10.1’

# 查看整个运行环境：加载的包及版本
> sessionInfo()
R version 3.5.1 (2018-07-02)
Platform: x86_64-pc-linux-gnu (64-bit)
Running under: Red Hat Enterprise Linux Server release 6.6 (Santiago)

Matrix products: default
BLAS: /Share/home/zhangqf5/software/R/3.5.1/lib64/R/lib/libRblas.so
LAPACK: /Share/home/zhangqf5/software/R/3.5.1/lib64/R/lib/libRlapack.so

locale:
 [1] LC_CTYPE=en_US.utf8       LC_NUMERIC=C
 [3] LC_TIME=en_US.utf8        LC_COLLATE=en_US.utf8
 [5] LC_MONETARY=en_US.utf8    LC_MESSAGES=en_US.utf8
 [7] LC_PAPER=en_US.utf8       LC_NAME=C
 [9] LC_ADDRESS=C              LC_TELEPHONE=C
[11] LC_MEASUREMENT=en_US.utf8 LC_IDENTIFICATION=C

attached base packages:
[1] parallel  stats4    stats     graphics  grDevices utils     datasets
[8] methods   base

other attached packages:
 [1] scater_1.10.1               ggplot2_3.1.0
 [3] SingleCellExperiment_1.4.1  SummarizedExperiment_1.12.0
 [5] DelayedArray_0.8.0          BiocParallel_1.16.6
 [7] matrixStats_0.54.0          Biobase_2.28.0
 [9] GenomicRanges_1.34.0        GenomeInfoDb_1.18.2
[11] IRanges_2.16.0              S4Vectors_0.20.1
[13] BiocGenerics_0.28.0

loaded via a namespace (and not attached):
 [1] beeswarm_0.2.3           tidyselect_0.2.5         xfun_0.4
 [4] reshape2_1.4.3           purrr_0.3.2              HDF5Array_1.10.1
 [7] lattice_0.20-38          rhdf5_2.26.2             colorspace_1.4-0
[10] htmltools_0.3.6          viridisLite_0.3.0        yaml_2.2.0
[13] rlang_0.3.1              pillar_1.3.1             glue_1.3.0
[16] withr_2.1.2              GenomeInfoDbData_1.2.0   plyr_1.8.4
[19] stringr_1.3.1            zlibbioc_1.14.0          munsell_0.5.0
[22] gtable_0.2.0             evaluate_0.12            knitr_1.21
[25] vipor_0.4.5              Rcpp_1.0.1               scales_1.0.0
[28] backports_1.1.3          XVector_0.22.0           gridExtra_2.3
[31] digest_0.6.18            stringi_1.2.4            dplyr_0.8.0.1
[34] grid_3.5.1               rprojroot_1.3-2          tools_3.5.1
[37] bitops_1.0-6             magrittr_1.5             lazyeval_0.2.1
[40] RCurl_1.95-4.11          tibble_2.0.1             crayon_1.3.4
[43] pkgconfig_2.0.2          Matrix_1.2-16            DelayedMatrixStats_1.4.0
[46] ggbeeswarm_0.6.0         assertthat_0.2.0         rmarkdown_1.10
[49] viridis_0.5.1            Rhdf5lib_1.4.3           R6_2.3.0
[52] compiler_3.5.1
```
---

### 查看当前bioconductor的版本

有时候需要指定的bioconductor版本，才能正确的运行程序。因为不同的版本其安装的包的版本不同，如果不同版本差异很大，经常是容易报错的。比如最近遇到的这个：[scRNAseq_workflow_benchmark](https://github.com/Novartis/scRNAseq_workflow_benchmark)，其强调说的`bioconductor`版本是3.5，如果是其他版本，跑出来报错，尤其是通过其安装的包`scater`，差异很大，已知运行不成功。

```R
> tools:::.BioC_version_associated_with_R_version() 
[1] ‘3.5’
```

---

### 从bioconductor安装指定版本的包

比如可能要安装低版本的包，需要先在bioconductor中找到对应的，然后指定安装url：

```R
# 比如：安装低版本的scater
# 这里没有成功，因为还有几个其他依赖的包安装不成功
install_version('scater', version='1.4.0',repos = "https://bioconductor.org/packages/3.5/bioc")
```

---

### 从source文件安装包

```R
# 先下载源文件，一般是压缩的
install.packages('/Share2/home/zhangqf5/gongjing/software/mvoutlier_2.0.9.tar.gz', repos = NULL, type="source")
```

---

### 移除安装包

```R
# https://stat.ethz.ch/R-manual/R-devel/library/utils/html/remove.packages.html
remove.packages(pkgs, lib)
```

---

### 设置包的加载路径

在集群上，有的时候自己的目录下面安装了包，但是在系统调用的时候可能没有加载进来，可能是因为libpath中的路径存在问题，导致一些依赖包的顺序不对，所以不能正常的调用：

```R
# 系统路径是在前的
# 直接调用时，依赖的包是系统的R安装的
# 与用户的R版本不一致，出现问题
> .libPaths()
[1] "/Share/home/zhangqf5/R/x86_64-pc-linux-gnu-library/3.3"
[2] "/Share/home/zhangqf/usr/R-3.3.0/lib64/R/library"
[3] "/Share/home/zhangqf/usr/R-3.3.2/lib64/R/library"
[4] "/Share/home/zhangqf5/software/R/3.5.1/lib64/R/library"
> library(mvoutlier)
Loading required package: sgeostat
Error: package or namespace load failed for ‘mvoutlier’:
 package ‘rrcov’ was installed by an R version with different internals; it needs to be reinstalled for use with this R version
>


# 更换lib顺序，使得用户的libpath优先调用
> .libPaths("/Share/home/zhangqf5/software/R/3.5.1/lib64/R/library")
>
> library(mvoutlier)
sROC 0.1-2 loaded
>
```

---

### 移除已经导入的包

有时候会遇到这种问题，先导入了module1，module1默认是导入module2(v1)的，接下来要导入module3，module3是依赖于module2(v2)的，因为v2和v1版本不同，所以使得不能正常导入module2。此时可以先把module1 deattach掉，然后专门的导入module2：

比如下面先导入了模块`m6ALogisticModel`，其默认导入`ggplot2`，接下来导入`caret`，但是其依赖的`ggplot2`版本不同，所以导入时报错：

```R
> library(caret)
Loading required package: ggplot2
Error in value[[3L]](cond) :
  Package ‘ggplot2’ version 3.1.0 cannot be unloaded:
 Error in unloadNamespace(package) : namespace ‘ggplot2’ is imported by ‘m6ALogisticModel’ so cannot be unloaded
```

可以先将`m6ALogisticModel`移除（参考[这里](https://stackoverflow.com/questions/6979917/how-to-unload-a-package-without-restarting-r)），再导入:

```R
> detach("package:m6ALogisticModel", unload=TRUE)
> library(caret)
Loading required package: ggplot2
```

---

### df中取不含某列列名的其他列

参考[这里](https://stackoverflow.com/questions/12868581/list-all-column-except-for-one-in-r)：

```R
# 提取training df中列名称不为class的其他列
Training.predictor.Gendata = training[,!names(training) %in% c("class")]
```