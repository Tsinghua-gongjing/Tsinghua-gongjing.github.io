---
layout: post
category: "machinelearning"
title:  "sklearn: 数据集加载"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

#### 1. 数据集

* 类：`sklearn.datasets`
* 通用数据
* 生成数据
* 获取真实数据

#### 2. 通用数据集

三种API接口：

* `loader`：加载小的标准数据集
* `fetchers`：下载大的真实数据集
* `generate functions`：生成受控的合成数据集

#### 3. 通用标准数据集

##### boston house-prices 

[![dataset_boston_house_price.jpeg](https://i.loli.net/2019/07/21/5d333b955114276295.jpeg)](https://i.loli.net/2019/07/21/5d333b955114276295.jpeg)

##### Iris 

[![dataset_iris.jpeg](https://i.loli.net/2019/07/21/5d333e668991227533.jpeg)](https://i.loli.net/2019/07/21/5d333e668991227533.jpeg)

##### diabetes 

[![dataset_diabetes.jpeg](https://i.loli.net/2019/07/21/5d3344b405a5696367.jpeg)](https://i.loli.net/2019/07/21/5d3344b405a5696367.jpeg)

##### digitals 

[![dataset_digitals.jpeg](https://i.loli.net/2019/07/21/5d3344c8e2b1413841.jpeg)](https://i.loli.net/2019/07/21/5d3344c8e2b1413841.jpeg)

##### linnerud 

[![dataset_linnerrud.jpeg](https://i.loli.net/2019/07/21/5d3344d9cce7f72487.jpeg)](https://i.loli.net/2019/07/21/5d3344d9cce7f72487.jpeg)

##### wine 

[![dataset_wine.jpeg](https://i.loli.net/2019/07/21/5d334545274a919488.jpeg)](https://i.loli.net/2019/07/21/5d334545274a919488.jpeg)

##### breast cancer 

[![dataset_breast_cancer.jpeg](https://i.loli.net/2019/07/21/5d3344ed9352361529.jpeg)](https://i.loli.net/2019/07/21/5d3344ed9352361529.jpeg)

#### 4. 真实数据

##### The Olivetti faces dataset

* description: a set of face images taken between April 1992 and April 1994 at AT&T Laboratories Cambridge
* classes: 40
* samples: 400
* dimensionality: 4096
* features: real, [0,1]

##### The Olivetti faces dataset

* description: comprises around 18000 newsgroups posts on 20 topics split in two subsets: one for training (or development) and the other one for testing (or for performance evaluation). 
* classes: 20
* samples: 18846
* dimensionality: 1
* features: text

##### The Labeled Faces in the Wild face recognition dataset

* description: a collection of JPEG pictures of famous people collected over the internet 
* classes: 5479
* samples: 13233
* dimensionality: 5828
* features: real, [0,255]

##### Forest covertypes

* description: 30×30m patches of forest in the US, collected for the task of predicting each patch’s cover type 
* classes: 7
* samples: 581012
* dimensionality: 54
* features: int

##### RCV1 dataset

* description: Reuters Corpus Volume I (RCV1) is an archive of over 800,000 manually categorized newswire stories made available by Reuters, Ltd. for research purposes 
* classes: 103
* samples: 804414
* dimensionality: 47236
* features: real, [0,1]

##### Kddcup 99 dataset

* description: created by processing the tcpdump portions of the 1998 DARPA Intrusion Detection System (IDS) Evaluation dataset, created by MIT Lincoln Lab
* classes: 总共分为了4个小类的数据，具体参见[这里](https://scikit-learn.org/stable/datasets/index.html#kddcup-99-dataset)
* samples: 4898431
* dimensionality: 41
* features: discrete (int) or continuous (float)

##### California Housing dataset

* description: Reuters Corpus Volume I (RCV1) is an archive of over 800,000 manually categorized newswire stories made available by Reuters, Ltd. for research purposes 
* samples: 20640
* dimensionality: 8
* features: real
* 和波斯顿房价数据的区别：这个是基于房屋本身属性的，而那个是基于城市地区属性的，这个其实更接近现实一点。

~~~
MedInc median income in block
HouseAge median house age in block
AveRooms average number of rooms
AveBedrms average number of bedrooms
Population block population
AveOccup average house occupancy
Latitude house block latitude
Longitude house block longitude
~~~

#### 5. 生成数据

##### 分类：单标签

函数：`make_blobs`
函数：`make_classification`

##### 分类：多标签

函数：`make_multilabel_classification`

##### 二分聚类

函数：`make_biclusters`，Generate an array with constant block diagonal structure for biclustering.
函数：`make_checkerboard`，Generate an array with block checkerboard structure for biclustering.

##### 回归生成器

函数：`make_regression`，产生的回归目标作为一个可选择的稀疏线性组合的具有噪声的随机的特征

##### 流行学习生成器

函数：`make_s_curve`，生成S曲线数据集
函数：`make_swiss_roll`，生成swiss roll数据集

#### 6. 下载公开数据集：openml.org

* openml.org：是一个用于机器学习数据和实验的公共存储库，它允许每个人上传开放的数据集
* 函数：`sklearn.datasets.fetch_openml`

```python
>>> from sklearn.datasets import fetch_openml
>>> mice = fetch_openml(name='miceprotein', version=4)
>>> 

# 查看数据集的信息和属性
# DESCR：自由文本描述数据
# details：字典格式的元数据
>>> print(mice.DESCR)
**Author**: Clara Higuera, Katheleen J. Gardiner, Krzysztof J. Cios
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression) - 2015
**Please cite**: Higuera C, Gardiner KJ, Cios KJ (2015) Self-Organizing
Feature Maps Identify Proteins Critical to Learning in a Mouse Model of Down
Syndrome. PLoS ONE 10(6): e0129126...

>>> mice.details
{'id': '40966', 'name': 'MiceProtein', 'version': '4', 'format': 'ARFF',
'upload_date': '2017-11-08T16:00:15', 'licence': 'Public',
'url': 'https://www.openml.org/data/v1/download/17928620/MiceProtein.arff',
'file_id': '17928620', 'default_target_attribute': 'class',
'row_id_attribute': 'MouseID',
'ignore_attribute': ['Genotype', 'Treatment', 'Behavior'],
'tag': ['OpenML-CC18', 'study_135', 'study_98', 'study_99'],
'visibility': 'public', 'status': 'active',
'md5_checksum': '3c479a6885bfa0438971388283a1ce32'}
```

#### 7. 加载外部数据集

* 数据集已经准备好了，自行加载以输入模型
* 不同的工具包：`pandas.io`,`scipy.io`,`numpy`
* 杂项数据：`skimage.io`,`Imagio`,`scipy.misc.imread`,`scipy.io.wavfile.read`

### 参考

* [据集加载工具@sklearn 中文](https://sklearn.apachecn.org/#/docs/47?id=_6-%E6%95%B0%E6%8D%AE%E9%9B%86%E5%8A%A0%E8%BD%BD%E5%B7%A5%E5%85%B7)