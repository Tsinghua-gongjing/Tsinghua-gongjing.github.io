---
layout: post
category: "machinelearning"
title:  "特征提取"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

#### 1. sklearn特征提取

* 模块`sklearn.feature_extraction`：提取符合机器学习算法的特征，比如文本和图片
* 特征提取（extraction）vs 特征选择（selection）：前者将任意数据转换为机器学习可用的数值特征，后者应用这些特征到机器学习中。

#### 2. 从字典类型数据加载特征

* 类`DictVectorizer`可处理字典元素，将字典中的数组转换为模型可用的数组
* 字典：使用方便、稀疏
* 实现`one-of-K`、`one-hot`编码，用于分类特征

```python
>>> measurements = [
...     {'city': 'Dubai', 'temperature': 33.},
...     {'city': 'London', 'temperature': 12.},
...     {'city': 'San Francisco', 'temperature': 18.},
... ]

>>> from sklearn.feature_extraction import DictVectorizer
>>> vec = DictVectorizer()

# 直接应用于字典，将分类的进行独热编码，数值型的保留
>>> vec.fit_transform(measurements).toarray()
array([[  1.,   0.,   0.,  33.],
 [  0.,   1.,   0.,  12.],
 [  0.,   0.,   1.,  18.]])

>>> vec.get_feature_names()
['city=Dubai', 'city=London', 'city=San Francisco', 'temperature']
```

#### 3. 特征哈希：相当于一种降维

* 类：`FeatureHasher`
* 高速、低内存消耗的向量化方法
* 使用特征散列（散列法）技术
* 接受映射
* 可在文档分类中使用
* 目标：把原始的高维特征向量压缩成较低维特征向量，且尽量不损失原始特征的表达能力

* 哈希表：有一个哈希函数，实现键值的映射，哈希把不同的键散列到不同的块，但还是存在冲突，即把不同的键散列映射到相同的值。

* 下面两篇文章介绍了特征哈希在文本上的使用，但是还没没有完全看明白！！！
* [数据特征处理之特征哈希（Feature Hashing）](https://www.datalearner.com/blog/1051537932880901)
* [文本挖掘预处理之向量化与Hash Trick](https://www.cnblogs.com/pinard/p/6688348.html)

```python
# 使用词袋模型
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer=CountVectorizer()
corpus=["I come to China to travel", 
    "This is a car polupar in China",          
    "I love tea and Apple ",   
    "The work is to write some papers in science"] 
print vectorizer.fit_transform(corpus)
print vectorizer.fit_transform(corpus).toarray()
print vectorizer.get_feature_names()
```

~~~
(0, 16)	1
  (0, 3)	1
  (0, 15)	2
  (0, 4)	1
  (1, 5)	1
  (1, 9)	1
  (1, 2)	1
  (1, 6)	1
  (1, 14)	1
  (1, 3)	1
  (2, 1)	1
  (2, 0)	1
  (2, 12)	1
  (2, 7)	1
  (3, 10)	1
  (3, 8)	1
  (3, 11)	1
  (3, 18)	1
  (3, 17)	1
  (3, 13)	1
  (3, 5)	1
  (3, 6)	1
  (3, 15)	1
[[0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 2 1 0 0]
 [0 0 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 0]
 [1 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 0 0 1 1 0 1 0 1 1 0 1 0 1 0 1 1]]
[u'and', u'apple', u'car', u'china', u'come', u'in', u'is', u'love', u'papers', u'polupar', u'science', u'some', u'tea', u'the', u'this', u'to', u'travel', u'work', u'write']
~~~

```python
# 使用hash技巧，将原来的19维降至6维
from sklearn.feature_extraction.text import HashingVectorizer 
vectorizer2=HashingVectorizer(n_features = 6,norm = None)
print vectorizer2.fit_transform(corpus)
print vectorizer2.fit_transform(corpus).toarray()
```

~~~
  (0, 1)	2.0
  (0, 2)	-1.0
  (0, 4)	1.0
  (0, 5)	-1.0
  (1, 0)	1.0
  (1, 1)	1.0
  (1, 2)	-1.0
  (1, 5)	-1.0
  (2, 0)	2.0
  (2, 5)	-2.0
  (3, 0)	0.0
  (3, 1)	4.0
  (3, 2)	-1.0
  (3, 3)	1.0
  (3, 5)	-1.0
[[ 0.  2. -1.  0.  1. -1.]
 [ 1.  1. -1.  0.  0. -1.]
 [ 2.  0.  0.  0.  0. -2.]
 [ 0.  4. -1.  1.  0. -1.]]
~~~

#### 4. 文本特征提取：话语表示

* 文本提取特征常见方法（词袋模型bag-of-words，BoW）：
	- 令牌化（tokenizing）：分出可能的单词赋予整数id
	- 统计（counting）：每个词令牌在文档中的出现次数
	- 标准化（normalizing）：赋予权重
* 特征：每个单独的令牌发生频率
* 样本：每个文档的所有令牌频率向量
* 向量化：将文本文档集合转换为数字集合特征向量
* 词集模型：只考虑单词出现与否，不考虑频率
* 稀疏：得到的文本矩阵很多特征值为0，通常大于99%
* **类：`CountVectorizer`，词切分+频数统计，用法参见上面的例子**

#### 5. Tf-idf项加权

* 有的词出现很多次，其实无用，比如the，is，a
* 重新计算特征权重：tf-idf变换
* Tf：term frequency，术语频率；
* idf：inverse document frequency，转制文档频率
* 公式：$$tf-idf(t,d) = tf(t,d) \times idf(t)$$

* 类：`TfidfTransformer`
* 类：`TfidfVectorizer`，组合CountVectorizer+TfidfTransformer

```python
>>> from sklearn.feature_extraction.text import TfidfTransformer
>>> transformer = TfidfTransformer(smooth_idf=False)
>>> transformer   
TfidfTransformer(norm=...'l2', smooth_idf=False, sublinear_tf=False,
 use_idf=True)
 
# 直观分析count
# 第一个词在所有文档都出现，可能不重要
# 另外两个词，出现不到50%，可能具有代表性
>>> counts = [[3, 0, 1],
...           [2, 0, 0],
...           [3, 0, 0],
...           [4, 0, 0],
...           [3, 2, 0],
...           [3, 0, 2]]
...
>>> tfidf = transformer.fit_transform(counts)
>>> tfidf                         
<6x3 sparse matrix of type '<... 'numpy.float64'>'
    with 9 stored elements in Compressed Sparse ... format>

>>> tfidf.toarray()                        
array([[0.81940995, 0.        , 0.57320793],
       [1.        , 0.        , 0.        ],
       [1.        , 0.        , 0.        ],
       [1.        , 0.        , 0.        ],
       [0.47330339, 0.88089948, 0.        ],
       [0.58149261, 0.        , 0.81355169]])
       
       
# TfidfVectorizer
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> vectorizer = TfidfVectorizer()
>>> vectorizer.fit_transform(corpus)
...                                
<4x9 sparse matrix of type '<... 'numpy.float64'>'
 with 19 stored elements in Compressed Sparse ... format>
```

### 参考

* [sklearn 中文](https://sklearn.apachecn.org/#/docs/39?id=_52-%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96)