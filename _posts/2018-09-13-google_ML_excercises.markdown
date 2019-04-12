---
layout: post
category: "machinelearning"
title:  "Google ML excercises"
tags: [python, machine learning]
---

谷歌机器学习课程对应的[练习](https://developers.google.com/machine-learning/crash-course/exercises)。

## 预热

1. [Hello World](): 正常导入TF模块，其提供的notebook都是基于python3的，所以最好安装anaconda3，然后安装对应的tensorflow。
2. [TensorFlow编程概念]():
    - 张量(tensors)：任意维度的数组。可作为常量或变量存储在图中。
    - 标量：零维数组（零阶张量）；矢量：一维数组（一阶张量）；矩阵：二维数组（二阶张量）
    - 指令：创建、销毁和操控张量
    - 图（计算图、数据流图）：图数据结构，其节点是指令，边是张量。
    - 会话
    - 1）将常量、变量和指令整合到一个图中；2）在一个会话中评估这些常量、变量和指令。
3. [创建和操控张量]()：

- 矢量加法：类似于python里面的数组（numpy数组），比如元素级加法、元素翻倍等。 函数：`tf.add(x,y)`
- 广播：**元素级运算中的较小数组会增大到与较大数组具有相同的形状**。一般数学上只支持形状相同的张量进行元素级的运算，但是TF中借鉴Numpy的广播做法。

```python
twos = tf.constant([2, 2, 2, 2, 2, 2], dtype=tf.int32)
primes_doubled = primes * twos
print("primes_doubled:", primes_doubled)

# 输出值包含值和形状(shape)、类型信息
primes_doubled: tf.Tensor([ 4  6 10 14 22 26], shape=(6,), dtype=int32)
```

- 矩阵乘法：第一个举证的**列数**必须等于第二个矩阵的**行数**。函数：`tf.matmul`
- 张量变形：矩阵运算限制了矩阵的形状，因此需要频繁变化，可调用`tf.reshape`函数，改变形状或者阶数。
- 变量初始化和赋值：如果定义的是变量，需要进行初始化，这个值之后是可以更改(函数：`tf.assign`)的。

```python
v = tf.contrib.eager.Variable([3])
print(v)
tf.assign(v, [7])
print(v)

[3]
[7]
```

模拟投两个骰子10次，10x3的张量，第三列是前两列的和：
   - `random_uniform` ：随机选取
   - `tf.concat` ：合并张量

```python
die1 = tf.contrib.eager.Variable(
    tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))
die2 = tf.contrib.eager.Variable(
    tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))

dice_sum = tf.add(die1, die2)
resulting_matrix = (values=[die1, die2, dice_sum], axis=1)

print(resulting_matrix.numpy())

[[5 1 6]
 [4 5 9]
 [1 6 7]
 [2 3 5]
 [1 2 3]
 [5 4 9]
 [3 5 8]
 [3 2 5]
 [3 1 4]
 [6 3 9]]
```

4. [Pandas简介]()：
   - DataFrame：数据表格
   - Series：单一列
   - 利用索引随机化数据
   - 重建索引不包含时，会添加新行，并以`NaN`填充
   
```python
cities

City name	Population	Area square miles	Population density
0	San Francisco	852469	46.87	18187.945381
1	San Jose	1015785	176.53	5754.177760
2	Sacramento	485199	97.92	4955.055147

# 随机化index，然后重建索引
cities.reindex(np.random.permutation(cities.index))

# 索引超出范围不报错，新建行
cities.reindex([0, 4, 5, 2])
```
     
---

## 问题构建

[问题构建(Framing)](https://developers.google.com/machine-learning/crash-course/framing/check-your-understanding)理解：

1. 【监督式学习】：假设您想开发一种监督式机器学习模型来预测指定的电子邮件是“垃圾邮件”还是“非垃圾邮件”。以下哪些表述正确？
  - `有些标签可能不可靠。`
  - `未标记为“垃圾邮件”或“非垃圾邮件”的电子邮件是无标签样本。`
  - 我们将使用无标签样本来训练模型。
  - 主题标头中的字词适合做标签。
2. 【特征和标签】：假设一家在线鞋店希望创建一种监督式机器学习模型，以便为用户提供合乎个人需求的鞋子推荐。也就是说，该模型会向小马推荐某些鞋子，而向小美推荐另外一些鞋子。以下哪些表述正确？
  - 鞋的美观程度是一项实用特征。
  - 用户喜欢的鞋子是一种实用标签。
  - `“用户点击鞋子描述”是一项实用标签。`
  - `鞋码是一项实用特征。`

---

## 深入了解机器学习

[均方误差](https://developers.google.com/machine-learning/crash-course/descending-into-ml/check-your-understanding):

[![MSE.jpeg](https://i.loli.net/2019/04/12/5cb0275977e78.jpeg)](https://i.loli.net/2019/04/12/5cb0275977e78.jpeg)

---

## 降低损失

1. [优化学习速率](https://developers.google.com/machine-learning/crash-course/fitter/graph)：
   - Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect the loss gradient（[ref](https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10)）. 所以需要不断的更新权重值，看更新后的损失值是否变小，所以横轴是权重，纵轴是损失值。
   - **新权重 = 旧权重 — 学习速率 * 梯度**
   - 在训练模型时，需要最小化损失函数，通常沿着梯度下降的方向。能否找到、多久找到损失函数的最小值，需要进行尝试，一个重要的参数就是学习速率(learning rate)。`学习速率*梯度`就是每次更新的数值。
   - 这个页面以互动的方式，调节学习速率，看损失函数（二项函数）经过多少步能到达最小损失处。
   - 这个图也展示了不同的学习速率可能导致的结果：太小则收敛太慢，太大则很难找到最小值处：
   - [![](http://p0.ifengimg.com/pmop/2018/0305/4BC4FD06768FB8963EF62E8A19344D7B329C3899_size37_w1080_h419.jpeg)](http://p0.ifengimg.com/pmop/2018/0305/4BC4FD06768FB8963EF62E8A19344D7B329C3899_size37_w1080_h419.jpeg)
2. 【[批量大小](https://developers.google.com/machine-learning/crash-course/reducing-loss/check-your-understanding)】：基于大型数据集执行梯度下降法时，以下哪个批量大小可能比较高效？
   - `小批量或甚至包含一个样本的批量 (SGD)。`
   - 全批量。
3. [学习速率和收敛](https://developers.google.com/machine-learning/crash-course/reducing-loss/playground-exercise): 学习速率越小，收敛所花费的时间越多。在这个练习中，可以尝试不同的学习速率，看能否收敛以及收敛所需的迭代次数。

---

## 使用TensorFlow的起始步骤

1. [使用 TensorFlow 的起始步骤]():

任务：基于单个输入特征预测各城市街区的房屋价值中位数

```python
# 准备feature和target数据，这里只用一个特征
my_feature = california_housing_dataframe[["total_rooms"]]
feature_columns = [tf.feature_column.numeric_column("total_rooms")]
targets = california_housing_dataframe["median_house_value"]

# 配置训练时的优化器
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# 配置模型
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
    """
  
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
    
# 训练模型
_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=100
)

# 评估模型
prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
predictions = linear_regressor.predict(input_fn=prediction_input_fn)
predictions = np.array([item['predictions'][0] for item in predictions])

mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

Mean Squared Error (on training data): 56367.025
Root Mean Squared Error (on training data): 237.417
```

模型效果好吗？通过`RMSE（特性：它可以在与原目标相同的规模下解读）`，直接与原来target的最大最小值，进行比较。在这里，误差跨越了目标范围的一半，所以需要优化模型减小误差。

```python
Min. Median House Value: 14.999
Max. Median House Value: 500.001
Difference between Min. and Max.: 485.002
Root Mean Squared Error: 237.417
```

问题：有适用于模型调整的标准启发法吗？

- 不同超参数的效果取决于数据，**不存在必须遵循的规则，需要自行测试和验证**。
- 好的模型应该看到训练误差应该稳步减小（刚开始是急剧减小），最终收敛达到平衡。
- 如果训练误差减小速度过慢，则提高学习速率也许有助于加快其减小速度。
- 如果训练误差变化很大，尝试降低学习速率。
- 较低的学习速率和较大的步数/较大的批量大小通常是不错的组合。
- 批量大小过小也会导致不稳定情况。不妨先尝试 100 或 1000 等较大的值，然后逐渐减小值的大小，直到出现性能降低的情况。

2. [合成特征和离群值]()：
  - 比如预测平均房价时把`total_rooms/population`作为一个特征，表示的是平均每个人有多少个房间，这个是可取的。
  - 识别离群值：
     - 1）画target和prediction的散点图，看是否在一条直线上
     - 2）画训练所用特征数据的分布图，看是否存在极端值
  - 截取离群值：设上下限值，超过的设置为对应的值 `clipped_feature = df["feature_name"].apply(lambda x: max(x, 0))`，再重新训练，看是否效果变好了。

---

## 训练集和测试集

[训练集和测试集](https://developers.google.com/machine-learning/crash-course/training-and-test-sets/playground-exercise)：调整训练集和测试集的相对比例，查看模型的效果。不同的数据集需要不同的比例，这个得通过尝试才能确定。

如何确定各data set之间的比例，没有统一的标准，取决于不同的条件 [About Train, Validation and Test Sets in Machine Learning](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7)：

- 取决于样本数量。
- 取决于所训练的模型。比如参数很多，可能需要很多数据进行验证。

---

## 验证

1. [检验您的直觉：验证](https://developers.google.com/machine-learning/crash-course/validation/check-your-intuition): 我们介绍了使用测试集和训练集来推动模型开发迭代的流程。在每次迭代时，我们都会对训练数据进行训练并评估测试数据，并以基于测试数据的评估结果为指导来选择和更改各种模型超参数，例如学习速率和特征。这种方法是否存在问题?
  - `多次重复执行该流程可能导致我们不知不觉地拟合我们的特定测试集的特性。`
  - 完全没问题。我们对训练数据进行训练，并对单独的预留测试数据进行评估。
  - 这种方法的计算效率不高。我们应该只选择一个默认的超参数集，并持续使用以节省资源。

2. [拆分验证集]()：

[![split_data_needs_randomize.jpeg](https://i.loli.net/2019/04/12/5cb08cfd75e8c.jpeg)](https://i.loli.net/2019/04/12/5cb08cfd75e8c.jpeg)

---

## 表示法

1. [特征集](https://github.com/Tsinghua-gongjing/blog_codes/blob/master/notebooks/google_ML/feature_sets.ipynb)：
  - 相关矩阵（Pearson coefficient）：探索不同特征之间的相似性
  - 更好的利用纬度信息：直观的纬度与房价没有线性关系，但是有的峰值可能与特定地区有关 =》可以把纬度转换为这些特定地区的相对值。

---

## 特征组合

## 简化正则化

## 分类

## 稀疏正则化

## 神经网络简介

## 训练神经网络

## 多类别神经网络

## 嵌套

## 静态训练和动态训练

## 静态推理和动态推理

## 数据依赖关系



