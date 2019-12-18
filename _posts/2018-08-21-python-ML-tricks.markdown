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

