---
layout: post
category: "machinelearning"
title:  "Python machine learning tricks"
tags: [python, machine learning, tricks]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

### Extract clusters from seaborn clustermap [stackoverflow](https://stackoverflow.com/questions/27924813/extracting-clusters-from-seaborn-clustermap)

```python
### direct call from clustermap
df = pd.read_csv(txt, header=0, sep='\t', index_col=0)
fig, ax = plt.subplots(figsize=(25,40))
sns.clustermap(df, standard_scale=False, row_cluster=True, col_cluster=True, figsize=(25, 40), yticklabels=False)
savefn = txt.replace('.txt', '.cluster.png')
plt.savefig(savefn)
plt.close()

### calc linkage in hierarchy
df_array = np.asarray(df)
row_linkage = hierarchy.linkage(distance.pdist(df), method='average')
col_linkage = hierarchy.linkage(distance.pdist(df.T), method='average')

fig, ax = plt.subplots(figsize=(25,40))
sns.clustermap(df, row_linkage=row_linkage, col_linkage=col_linkage, standard_scale=False, row_cluster=True, col_cluster=True, figsize=(25, 40), yticklabels=False, method="average")
savefn = txt.replace('.txt', '.cluster.png')
plt.savefig(savefn)
plt.close()
```

[`hierarchy.fcluster`](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.cluster.hierarchy.fcluster.html) can be used to extract clusters based on max depth or cluster num as illustrated [here](https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/):

```python
### max depth
from scipy.cluster.hierarchy import fcluster
max_d = 50
clusters = fcluster(Z, max_d, criterion='distance')
clusters

array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

### user defined cluster number
k=2
fcluster(Z, k, criterion='maxclust')
```

Assign cluster id and plot [(stackoverflow)](https://stackoverflow.com/questions/48173798/additional-row-colors-in-seaborn-cluster-map):

```python
fcluster = hierarchy.fcluster(row_linkage, 10, criterion='maxclust')
lut = dict(zip(set(fcluster), sns.hls_palette(len(set(fcluster)), l=0.5, s=0.8)))
row_colors = pd.DataFrame(fcluster)[0].map(lut)

sns.clustermap(df, row_linkage=row_linkage, col_linkage=col_linkage, 
				  standard_scale=False, row_cluster=True, col_cluster=False, 
				  figsize=(50, 80), yticklabels=False, method="average", 
				  cmap=cmap, row_colors=[row_colors])
```

---

### 指定运行的CPU数目

在新版本的pytorch里面，在运行程序时可能默认使用1个CPU的全部核，导致其他用户不能正常使用，可通过指定参数`OMP_NUM_THREADS=1`设置运行时使用的核数，下面是加和不加时的CPU监视截图：

不指定：

[![20191218105451](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191218105451.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191218105451.png)

指定：

[![20191218110256](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191218110256.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191218110256.png)

可以看到，在指定时，只使用了1个核，此时的CPU使用率是100%左右；在不指定时，CPU使用率到达了3600%，此时默认使用了36个核。具体命令如下：

```bash
# CUDA_VISIBLE_DEVICES: 指定GPU核
# OMP_NUM_THREADS: 指定使用的CPU核数
time CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python $script_dir/main.py
```

---

### 数据并行时报错：NN module weights are not part of single contiguous chunk of memory

参考[这里](https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282)

```python
# 之前
out1, _ = self.lstm(x, (h0, c0)) 

# 之后
self.lstm.flatten_parameters()
out1, _ = self.lstm(x, (h0, c0)) 
```

---

### 为什么RNN/LSTM很慢

在最近搭建的LSTM模型中，即使使用上了GPU，运行花费时间还是很长。模型不是很大，一个卷积层+一个resblock+一个LSTM，序列长度为100，使用双向的LSTM，两层、hidden size是128，基本运行需要10+h小时的时间，每一个epoch花费接近3min。有一些关于为什么LSTM运行很慢的文章（比如知乎帖子[rnn为什么训练速度慢](https://www.zhihu.com/question/292024466)），可以参考一下，主要是：

* 每一个时间点的计算，依赖于上一时间点，不能并行
* 计算量大。计算复杂度：O(seq_len)，序列越长越明显，且RNN内部使用的是全连接结构。
* 每个时间点还有1个memory I/O的操作

最近有个SRU的模块，可以实现如CNN级别的速度训练RNN网络，在[github](https://github.com/asappresearch/sru)上star数目还挺高的，可以参考一下。

---

### 异常值处理：Tukey's Method

* 计算异常值步进，即1.5倍的四分位数范围
* 所有IQR外超过异常值步进的数据定义为异常值并去除

```python
for feature in log_data.keys():
    Q1 = np.percentile(log_data[feature], 25)
    Q3 = np.percentile(log_data[feature], 75)
    step = 1.5 * (Q3 - Q1)

    print("特征'{}'的异常值包括:".format(feature))
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])

outliers  = [95, 338, 86, 75, 161, 183, 154] # 选择需要删除的异常值index

good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True) #删除选择的异常值
```