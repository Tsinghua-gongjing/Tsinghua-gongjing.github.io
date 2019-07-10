---
layout: post
category: "machinelearning"
title:  "树回归算法"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## 概述

1. 决策树：将数据不断切分，直到目标变量相同或者不可分为止。是一种贪心算法，做出最佳选择，不考虑到达全局最优。
2. ID3建树：
   - 切分过于迅速，使用某特征进行分割之后，此特征不再使用。切割数目与特征值取值个数有关，比如取4个值，则分为4类。
   - 不能直接处理连续型特征：需事先进行转换，此转换可能破坏连续型变量的内在性质
   - 二元切分：设定特征阈值，分为大于此阈值的和小于此阈值的，进行切分。
3. CART(Classification and Regression Tree)：二元切分处理连续型变量
   - 可处理标称型【分类】或者连续型【回归】变量
   - 分类：采用基尼系数选择最优特征
   - 回归：平方误差和选取特征和阈值。回归又包含以下两类：
   - 回归树（regression tree）：每个节点包含单个值
   - 模型树（model tree）：每个节点包含一个线性方程 ![](https://www.yingjoy.cn/wp-content/uploads/2018/03/3-1.png)
4. **回归树**：
   - 叶子节点是（分段）常数值
   - 恒量数据一致性（误差计算）：1）求数据均值，2）所有点到均值的差值的平方，3）平方和。类似于方差，方差是平方误差的均值（均方差），这里是平方误差的总值（总方差）。
5. 树剪枝（tree pruning）：
   - 为什么剪枝？节点过多，模型对数据过拟合。如何判断是否过拟合？交叉验证！
   - 剪枝（purning）：降低决策树的复杂度来避免过拟合的过程。
   
   - 预剪枝（prepurning）：设定提前终止条件等
	- 对于某些参数很敏感，比如设定的：tolS容许的误差下降值，tolN切分的最少样本数。当有的特征其数值数量级发生变化时，可能很影响最后的回归结果。
   
   - 后剪枝（postpurning）：使用测试集和训练集
   - 不用用户指定参数，更加的友好，基本过程如下：
   - ```python
   基于已有的树切分测试数据
      如果存在任一子集是一棵树，则在该子集递归剪枝过程
      计算将当前两个叶节点合并后的误差
      计算不合并的误差
      如果合并误差会降低误差的话，就将叶节点合并
   ```
6. **模型树**：
   - 把叶节点设置为**分段线性函数**（piecewise linear），每一段可用普通的线性回归
   - 模型由多个线性片段组成，与回归树相比，叶节点的返回值是一个模型，而不是一个数值（目标值的均值）
   - 结果更易于理解，具有更高的预测准确度


## 实现

### Python源码版本

回归树的构建：

```python
# 在给定特征和阈值，将数据集进行切割
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0,mat1

# 递归调用creatTree函数，进行树构建
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree 
```

回归树的切分：

```python
def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

# leafType:这里是regLeaf回归树，目标变量的均值
# errType: 这里是目标变量的平方误差 * 样本数目=平方误差和
# tolS = ops[0]; tolN = ops[1] tolS容许的误差下降值，tolN切分的最少样本数（确保有的支数目不能过少）
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        # 这里是遍历要切分的feature的所有可能取值集合，而不是这个范围内的所有值？
        for splitVal in set(dataSet[:,featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split
```

回归树剪枝：

```python
def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
    
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print "merging"
            return treeMean
        else: return tree
    else: return tree
```

模型树：

```python
# 普通的线性回归，返回模型
def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

# 计算误差：预测值和真实值之间的平方误差
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))
```

### sklearn版本

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
boston = load_boston()
regressor = DecisionTreeRegressor(random_state=0)
cross_val_score(regressor, boston.data, boston.target, cv=10)

# array([ 0.61..., 0.57..., -0.34..., 0.41..., 0.75...,
#        0.07..., 0.29..., 0.33..., -1.42..., -1.77...])
```

## 参考

* 机器学习实战第9章
* [机器学习算法实践-树回归](https://pytlab.github.io/2017/11/03/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E5%AE%9E%E8%B7%B5-%E6%A0%91%E5%9B%9E%E5%BD%92/)





