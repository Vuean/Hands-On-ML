# 第二章 端到端的机器学习项目

本章将介绍一个端到端的项目案例。主要经历的步骤有：

1. 考虑大局

2. 获得数据

3. 从数据探索和可视化中洞察数据

4. 机器学习算法的数据准备

5. 选择并训练模型

6. 微调模型

7. 展示解决方案

8. 启动、监控和维护系统

## 2.1 使用真实数据

学习机器学习最好使用真实数据进行实验，而不仅仅是人工数据集。常见获取数据的地方有：

1. 流行的开放数据存储库

    - [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/)

    - [Kaggle datasets](https://www.kaggle.com/datasets)

    - [Amazon’s AWS datasets](http://aws.amazon.com/fr/datasets/)

2. 元门户站点（它们会列出开放的数据存储库）

    - [Data Portals](http://dataportals.org/)

    - [OpenDataMonitor](http://opendatamonitor.eu/)

    - [Quandl](http://quandl.com/)

3. 其他一些列出许多流行的开放数据存储库的页面：

    - [Wikipedia’s list of Machine Learning datasets](https://goo.gl/SJHN2k)

    - [Quora.com](http://goo.gl/zDR78y）)

    - [The datasets subreddit](https://www.reddit.com/r/datasets)

本章从StatLib库中选择了加州住房价格的数据集。该数据集基于1990年加州人口普查的数据。虽然不算是最新的数据，但是有很多可以学习的特质。出于教学目的，我们还特意添加了一个分类属性，并且移除了一些特征。

![图1_加州住房价格](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter2/%E5%9B%BE1_%E5%8A%A0%E5%B7%9E%E4%BD%8F%E6%88%BF%E4%BB%B7%E6%A0%BC.jpg "图1_加州住房价格")

## 2.2 考虑大局（look at the big picture）

目前需要做的是使用加州人口普查的数据建立起加州的房价模型。数据中有许多指标，诸如每个街区（区域）的人口数量、收入中位数、房价中位数等。

### 2.2.1 框架问题

1. 确定业务目标

    首先应该确定的是**业务目标是什么**？例如针对本例，该模型的输出（对一个区域房价中位数的预测）将会与其他信息数据一同传输给另一个机器学习系统，而这个下游系统将用来决策一个给定的区域是否值得投资。因为直接影响到收益，所以正确获得这个信息至关重要。

    ![图2_一个针对房地产投资的机器学习流水线](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter2/%E5%9B%BE2_%E4%B8%80%E4%B8%AA%E9%92%88%E5%AF%B9%E6%88%BF%E5%9C%B0%E4%BA%A7%E6%8A%95%E8%B5%84%E7%9A%84%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%B5%81%E7%A8%8B%E5%9B%BE.jpg "图2_一个针对房地产投资的机器学习流水线")

    **流水线（piplines）**：一个序列的数据处理组件称为一个数据流水线。流水线在机器学习系统中非常普遍，因为需要大量的数据操作和数据转化才能应用。

    组件之间通常是异步运行的。每个组件拉取大量的数据，然后进行处理，再将结果传输给
    另一个数据仓库。每个组件都很独立：**组件和组件之间的连接只有数据仓库**。这样一来，即使某个组件发生故障，它下游的组件还能使用前面的最后一次输出继续正常运行一段时间，使得**整体架构鲁棒性较强**。

2. 掌握前期解决方案基础

    然后需要掌握当前已有的解决方案，可以此作为参考。

3. 设计系统

    确定问题框架：选择监督学习、无监督学习还是强化学习？是分类任务、回归任务还是其他任务？应该使用批量学习还是在线学习技术？

    根据问题描述可知：这属于监督学习、回归任务且是多重回归问题（使用了多个特征）。最后，该问题中没有一个连续的数据流不断流进系统，所以不需要针对变化的数据做出特别调整，数据量也不是很大，不需要多个内存，所以简单的批量学习应该就能胜任。

### 2.2.2 选择性能指标

回归问题的典型性能指标是**均方根误差**（Root Mean Square Error, RMSE）。它给出了系统通常会在预测中产生多大误差，对于较大的误差，权重较高。

![图3_均方根误差RMSE](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter2/%E5%9B%BE3_%E5%9D%87%E6%96%B9%E6%A0%B9%E8%AF%AF%E5%B7%AERMSE.jpg "图3_均方根误差RMSE")

其中，m为要在其上测量RMSE的数据集中的实例数；x(i)为数据集中第i个实例的所有特征值（不包括标签）向量；y(i)为数据集中第i个实例的标签（该实例的期望输出值）。**X**为矩阵，包含数据集中所有实例的所有特征值（不包括标签），第i行为x(i)的转置；h为系统的预测函数，也成为**假设**；RMSE(X, h)是使用假设h在一组示例中测量的成本函数。

通常情况下回归任务的首选性能指标就是**均方根误差，RMSE**，但是针对较多异常区域的情况下，可考虑使用**平均绝对误差**（Mean Absolute Error, MAE）。

![图4_平均绝对误差MAE]()

RMSE和MAE都是测量两个向量（预测值向量和目标值向量）之间距离的方法。各种距离度量或范数是可能的：

- 计算平方和的根（RMSE）与欧几里得范数相对应：ℓ2；

- 计算绝对值之和（MAE）对应ℓ1范数，也称为”曼哈顿范数“；

- 一般而言，包含n个元素的向量v的ℓk范数定义为如下式所示，ℓ0给出了向量中的非零元素数量， ℓ∞给出向量中的最大绝对值。

    ![图5_k范数定义公式]()

- 范数指标越高，它越关注大值而忽略小值。这就是RMSE对异常值比MAE更敏感的原因。但是，当离群值呈指数形式稀有时（如钟形曲线），RMSE表现非常好，通常是首选。

### 2.2.3 假设检查

最后，列举和验证到目前为止（由你或者其他人）做出的假设，是一个非常好的习惯。确定输出结果可行有用。

## 2.3 获取数据

完整的Jupyter notebook可以通过https://github.com/ageron/handsonml2获得。

### 2.3.1 创建工作区

### 2.3.2 下载数据

在典型环境中，数据存储在关系型数据库里（或其他一些常用数据存储），并分布在多个表/文档/文件中。可通过创建一个小函数，从浏览器下载压缩包，解压缩提取文件。

获取数据函数如下：

```python
    import os
    import tarfile
    import urllib.request

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    HOUSING_PATH = os.path.join("datasets", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
        if not os.path.isdir(housing_path): # 判断路径是否为目录
            os.makedirs(housing_path)   # 递归创建目录
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
```

现在，每调用fetch_housing_data()，就会自动在工作区创建一个datasets/housing目录，下载housuing.tgz文件，并解压至当前文件夹。

然后，通过pandas加载数据，返回一个包含所有数据的pandas DataFrame对象：

```python
    import pandas as pd

    def load_housing_data(housing_path = HOUSING_PATH):
        csv_path = os.path.join(housing_path, "housing.csv")
        return pd.read_csv(csv_path)
```

### 2.3.3 快速查看数据结构

使用DataFrames的**head()方法，查看数据的前5行**（详见ipynb文件）：

```python
    housing = load_housing_data()
    housing.head()
```

可以看出，每一行代表一个区域，总共有10个属性：longitude	latitude、housing_median_age、total_rooms、total_bedrooms、population、households	、median_income、median_house_value、ocean_proximity。

通过**info()方法可以以快速获取数据集的简单描述，特别是总行数、每个属性的类型和非空值的数量**。

可注意到，数据总量为20640个实例，其中total_bedrooms仅有20433个非空值，即意味着存在207个实例缺失该部分数据。同时注意到，除ocean_proximity外，所有属性的类型均为float型。ocean_proximity属灵类型为object，可能为文本类型，且存在重复，因此该类型数据可能是一个分类属性，可**通过value_counts()函数，查看有多少种分类存在**。

```python
    housing["ocean_proximity"].value_counts()
```

通过**describe()方法，可以显示数值属性的摘要信息**。

```python
    housing.describe()
```

另一种快速了解数据类型的方法是绘制每个数值属性的直方图。直方图用来显示给定值范围（横轴）的实例数量（纵轴）。可以**通过hist()方法，绘制每个属性的直方图**。

其中，hist()方法依赖于Matplotlib，而Matplotlib又依赖于用户指定的图形后端才能在屏幕上完成绘制。所以在绘制之前，你需要先指定Matplotlib使用哪个后端。最简单的选择是使用Jupyter的神奇命令%matplotlib inline。它会设置Matplotlib从而使用Jupyter自己的后端，随后图形会在notebook上呈现。

```python
    %matplotlib inline  # only in a jupyter notebook
    import matplotlib.pyplot as plt
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()
```

从直方图中，可注意到：

1. 收入中位数可发现并不是用美元为的单位，而是万美元（经沟通后发现）。

2. 房龄与房价被设置了上限，而由于房价正是目标属性（标签），因此用该数据来训练可能会导致机器学习算法学习到的价格永远都不会超过整个限制。通常可以：

    - 对那些标签值被设置了上限的区域，重新收集标签值；

    - 将这些区域的数据从训练集中移除。

3. 这些属性值被缩放的程度各不相同。

4. 许多直方图都表现出重尾：图形在中位数右侧的延伸比左侧要远得多。这可能会导致某些机器学习算法难以检测模式。

### 2.3.4 创建测试集

