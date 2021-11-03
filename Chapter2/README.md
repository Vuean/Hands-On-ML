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

![图4_平均绝对误差MAE](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter2/%E5%9B%BE4_%E5%B9%B3%E5%9D%87%E7%BB%9D%E5%AF%B9%E8%AF%AF%E5%B7%AEMAE.jpg)

RMSE和MAE都是测量两个向量（预测值向量和目标值向量）之间距离的方法。各种距离度量或范数是可能的：

- 计算平方和的根（RMSE）与欧几里得范数相对应：ℓ2；

- 计算绝对值之和（MAE）对应ℓ1范数，也称为”曼哈顿范数“；

- 一般而言，包含n个元素的向量v的ℓk范数定义为如下式所示，ℓ0给出了向量中的非零元素数量， ℓ∞给出向量中的最大绝对值。

    ![图5_k范数定义公式](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter2/%E5%9B%BE5_k%E8%8C%83%E6%95%B0%E5%AE%9A%E4%B9%89%E5%85%AC%E5%BC%8F.jpg)

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

当本人来浏览测试集数据时，很可能会陷入某个看似有趣的测试数据模式，进而选择某个特殊的机器学习模式，从而会导致再使用测试集对泛化误差率进行估算时，估计结果将会过于乐观，该系统启动后表现将不如预期优秀，这称为**数据窥探偏误**（data snooping bias）。

理论上，创建测试集非常简单，只需要随机选择一些实例，通常是**数据集的20%**（如
果数据集很大，比例将更小）：

```python
    # 创建测试集
    # For illustration only. Sklearn has train_test_split()
    def split_train_test(data, test_ratio):
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]
```

更理想的，通常是每个实例都使用一个标识符来决定是否进入测试集，以确保稳定的训练-测试分割。

```python
    from zlib import crc32

    def test_set_check(identifier, test_ratio):
        return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32

    def split_train_test_by_id(data, test_ratio, id_column):
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_ : test_set_check(id_, test_ratio))
        return data.loc[~in_test_set], data.loc[in_test_set]
```

不幸的是，housing数据集没有标识符列。最简单的解决方法是使用行索引作为ID：

```python
    # 更简单的是使用行索引作为ID
    housing_with_id = housing.reset_index() # add an 'index' column
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
```

如果使用行索引作为唯一标识符，需要确保在数据集的末尾添加新数据，并且不会删除任何行。如果不能保证这一点，那么你可以尝试使用某个最稳定的特征来创建唯一标识符。例如，一个区域的经纬度肯定几百万年都不会变，你可以将它们组合成如下的ID。

```python
    # 以特定数据组合为id
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
```

Scikit-Learn提供了一些函数，可以通过多种方式将数据集分成多个子集。最简单的函数是`train_test_split()`，除了几个额外特征,它与前面定义的函数plit_train_test()几乎相同。首先，它也有random_state参数，可以像之前提到过的那样设置随机生成器种子；其次，可以把行数相同的多个数据集一次性发送给它，它会根据相同的索引将其分：

```python
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

当数据集足够大时，上述的**随机抽样**方法可能会导致明显的抽样偏差。更多时可能需要采用**分层抽样**。

假如已知，要预测房价中位数，收入中位数是一个非常重要的属性。于是需要在“收入”属性上，使得测试集能够代表整个数据集中各种不同类型的收入。首先，需要创建一个”收入类别“的属性，以收入中位数直方图为例：

```python
    housing["income_cat"] = pd.cut(housing["median_income"],     bins=[0.,1.5,3.0,4.5,6., np.inf],labels=[1, 2, 3, 4, 5])
    housing["income_cat"].hist()
```

通过收入直方图可以看出，大多数收入中位数值聚集在1.5～6（15000～60000美元）左右，但也有一部分远远超过了6万美元。**在数据集中，每一层都要有足够数量的实例，这一点至关重要，不然数据不足的层，其重要程度很有可能会被错估。**

上述，利用pd.cut()函数，创建5个收入类别的属性，0~1.5是类别1，1.5~3是类别2，以此类推。

然后再根据收入类别进行分层抽样：

```python
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    strat_test_set["income_cat"].value_counts() / len(strat_test_set)
```

通过比较，可得到在三种不同的数据集（完整数据集、分层抽样的测试集、纯随机抽样的测试集）中收入类别比例分布。

```python
    # 查看完整数据集、分层抽样的测试集、纯随机抽样的测试集中收入类别比例分布：
    def income_cat_proportions(data):
        return data["income_cat"].value_counts() / len(data)

    train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)

    compare_props = pd.DataFrame({
        "Overall": income_cat_proportions(housing),
        "Stratified":income_cat_proportions(strat_test_set),
        "Random":income_cat_proportions(test_set),
    }).sort_index()

    compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
    compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
```

可以看出，分层抽样的测试集中的比例分布与完整数据集中的分布几乎一致，而纯随机抽样的测试集结果则是有偏的。

最后删除income_cat属性，将数据恢复原样：

```python
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
```

## 2.4 从数据探索和可视化中获得洞见

在此阶段，将从快速浏览数据的基础上更深入地了解数据。

当如果训练集非常庞大，可以抽样一个探索集，这样后面的操作更简单快捷一些。首先，创建一个数据副本，保证可以随便尝试而不损害训练集：

```python
    # 创建数据副本
    housing = strat_train_set.copy()
```

### 2.4.1 将地理数据可视化

由于存在地理位置信息（经度和纬度），因此建立一个各区域的分布图以便于可视化数据是一个很好的想法：

```python
    # 可视化地理位置信息
    housing.plot(kind="scatter", x = "longitude", y = "latitude")
```

![图6_数据的地理散点图]()

将alpha选项设置为0.1（**调节透明度**），可以更清楚地看出高密度数据点的位置：

```python
    # 美化图片，设置透明度
    housing.plot(kind="scatter", x="longitude", y="latitude",alpha=0.1)
```

![图7_突出高密度区域的地理散点图]()

从优化后的可视化图中可以清楚地分辨出高密度区域，如湾区、洛杉矶和圣地亚哥附近等。进一步地，再通过可视化工具，展现房价信息。使用jet工具来预定义颜色表（选项cmap）
来进行可视化，颜色范围从蓝（低）到红（高）。

```python
    # 房价可视化
    housing.plot(kind = "scatter",x="longitude", y="latitude", alpha=0.4,
                            s=housing["population"]/100,label="population",figsize=(10,7),
                            c = "median_house_value", cmap=plt.get_cmap("jet"),colorbar=True,)
    plt.legend()
```

![图8_人口-收入房价图]()

从上图中可以看出，房价与地理位置（例如靠海）、人口密度息息相关。一个通常很有用的方法是使用**聚类算法**来检测主集群，然后再为各个集群中心添加一个新的衡量邻近距离的特征。海洋邻近度可能就是一个很有用的属性，不过在北加州，沿海地区的房价并不是太高，所以这个简单的规则也不是万能的。

### 2.4.2 寻找相关性

针对少量数据集，可以使用corr()方法轻松计算出每对属性之间的标准相关系数（也称为皮尔逊r）：

```python
    # 使用corr()方法轻松计算出每对属性之间的标准相关系数
    corr_matrix = housing.corr()

    # 查看每个属性与房价的相关性：
    corr_matrix["median_house_value"].sort_values(ascending=False)

    # 结果：
    median_house_value    1.000000
    median_income         0.687160
    total_rooms           0.135097
    housing_median_age    0.114110
    households            0.064506
    total_bedrooms        0.047689
    population           -0.026920
    longitude            -0.047432
    latitude             -0.142724
    Name: median_house_value, dtype: float64
```

相关系数的范围从-1变化到1。越接近1，表示有越强的正相关。当系数接近于-1时，表示有较强的负相关。**相关系数仅测量线性相关性（“如果x上升，则y上升/下降”）。所以有可能彻底遗漏非线性相关性（例如“如果x接近于0，则y会上升”）**。

还有一种方法可以检测属性之间的相关性，就是使用pandas的scatter_matrix函数，它会绘制出每个数值属性相对于其他数值属性的相关性。现在我们有11个数值属性，可以得到11^2=121个图像，篇幅原因无法完全展示，这里我们仅关注那些与房价中位数属性最相关的，可算作是最有潜力的属性：

```python
    # 绘制每个数值属性相对于其他数值属性的相关性
from pandas.plotting import scatter_matrix

    attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]

    scatter_matrix(housing[attributes], figsize=(12,8))

    # 如果pandas绘制每个变量对自身的图像，那么主对角线（从左上到右下）将全都是直线，
    # 这样毫无意义。所以取而代之的方法是，pandas在这几个图中显示了每个属性的直方图
```

![图9_属性间散布矩阵图]()

从图中可以看出，最有潜力能够预测房价中位数的属性是收入中位数，放大观测其相关性：

```python

```

![图10_收入中位数与房价中位数]()

从图中可以看出，首先，二者的相关性确实很强，可以清楚地看到上升的趋势，并且点也不是太分散。其次，前面提到过50万美元的价格上限在图中是一条清晰的水平线，不过除此之外，上图还显示出几条不那么明显的直线：45万美元附近有一条水平线，35万美元附近也有一条，28万美元附近似乎隐约也有一条，再往下可能还有一些。为了避免你的算法学习之后重现这些怪异数据，你可能会尝试删除这些相应区域。

### 2.4.3 实验不同属性的组合

在准备开始给机器学习算法输入数据之前，可能识别出了一些异常数据，**需要提前清理掉**。

在准备给机器学习算法输入数据之前，最后一件需要做的事情就是：尝试各种属性的组合。例如，如果不知道一个区域有多少个家庭，那么知道一个区域的“房间总数”也没什么用。而真正想要知道的是一个家庭的房间数量。同样，单看“卧室总数”这个属性本身也没什么意义，可能想拿它和“房间总数”来对比，或者拿来同“每个家庭的人口数”这个属性组合似乎也挺有意思。来试着创建这些新属性：

```python
    # 组合新属性
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    # 查看相关矩阵
    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending = False)
```

从结果可以看出，新属性bedrooms_per_room较之“房间总数”或“卧室总数”与房价中位数的相关性都要高得多。显然，卧室/房间比例更低的房屋往往价格更贵。

## 2.5 机器学习算法的数据准备

在为机器学习算法准备数据的过程中，可自行编写函数来执行，这样一来可以实现：

- 在任何数据集上轻松重现这些转换；

- 建立起一个转换函数的函数库；

- 在实时系统中使用这些函数来转换新数据，再输入给算法；

- 尝试多种转换方式，查看哪种转换的组合效果最佳。

再次，回到一个干净的训练集（再次复制strat_train_set），然后将预测器和标签分开。drop()函数会建一个数据副本，但是不影响原始数据。

```python
    # 准备训练集：
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
```

### 2.5.1 数据清洗

大部分的机器学习算法无法在缺失的特征上工作，所以我们要创建一些函数来辅助它。在数据展示阶段，已经注意到total_bedrooms属性有部分值缺失，所以必须要解决它。有以下三种选择：

- 放弃这些区域；

- 放弃整个属性；

- 将确实的值设置为某个值（0、平均数或中位数等）

通过DataFrame的dropna()、drop()和fillna()方法，可以轻松完成这些操作：

```python
    # 处理缺失数据

    housing.dropna(subset=["total_bedrooms"])       # 放弃相应区域

    housing.drop("total_bedrooms", axis=1)              # 放弃整个属性

    median = housing["total_bedrooms"].median()     # 设置为某个值（0、平均数、中位数等）
    housing["total_bedrooms"].fillna(median, inplace=True)
```

Scikit-Learn提供了一个非常容易上手的类来处理缺失值：`SimpleImputer`。使用方法如下：首先，需要创建一个SimpleImputer实例，指定要用属性的中位数值替换该属性的缺失值：

```python
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
```

由于中位数值只能在数值属性上计算，所以我们需要创建一个没有文本属性ocean_proximity的数据副本：

```python
    housing_num = housing.drop("ocean_proximity", axis=1)
```

使用fit()方法将imputer实例适配到训练数据：

```python
    imputer.fit(housing_num)
```

这里imputer仅仅只是计算了每个属性的中位数值，并将结果存储在其实例变量statistics_中。虽然只有total_bedrooms这个属性存在缺失值，但是我们无法确认系统启动之后新数据中是否一定不存在任何缺失值，所以稳妥起见，还是将imputer应用于所有的数值属性：

```python
    >>> imputer.statistics_
    array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    , 408.    ,    3.5409])

    >>> housing_num.median().values
    array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    , 408.    ,    3.5409])
```

使用imputer将缺失值替换成中位数值从而完成训练集转换：

```python
    X = imputer.transform(housing_num)
```

结果是一个包含转换后特征的NumPy数组。如果想将它放回pandas DataFrame，也很简单：

```python
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
```

### 2.5.2 处理文本和分类属性

处理完数值属性后，再处理文本属性。在本例中，仅ocean_proximity属性为文本类，并且通过观察可知该属性的取值是有限个可能的取值，为分类属性。通常可以将分类属性转换为数字，可通过Scikit-Learn的OrdinalEncoder类：

```python
    from sklearn.preprocessing import OrdinalEncoder
    oridinal_encoder = OrdinalEncoder()
    housing_cat_encoded = oridinal_encoder.fit_transform(housing_cat)
    housing_cat_encoded[:10]
```

使用Categories_实例变量获取类别列表。这个列表包含每个类别属性的一维数组（在这种情况下，这个列表包含一个数组，因为只有一个类别属性）：

```python
    # 使用Categories_实例变量获取类别列表
    oridinal_encoder.categories_
```

但是这种表征方式产生的一个问题是，机器学习算法会认为两个相近的值比两个离得较远的值更为相似一些。新的属性有时候称为**哑（dummy）属性**。Scikit-Learn提供了一个OneHotEncoder编码器，可以将整数类别值转换为独热向量。我们用它来将类别编码为独热向量。

独热编码即 One-Hot 编码，又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。

[独热编码One-Hot Encoding](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter2/%E7%8B%AC%E7%83%AD%E7%BC%96%E7%A0%81One-Hot-Encoding.ipynb)

```python
    from sklearn.preprocessing import OneHotEncoder
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    housing_cat_1hot
```

且该函数输出得是一个**Scipy稀疏矩阵**，而不是一个NumPy数组。因为在完成独热编码之后，得到一个几千列的矩阵，并且全是0，每行仅有一个1。占用大量内存来存储0是一件非常浪费的事情，因此稀疏矩阵选择仅存储非零元素的位置，并依旧可以像使用一个普通的二维数组那样来使用它。如果需要将稀疏矩阵转换成一个（密集的）NumPy数组，只需要调用toarray()方法即可：

```python
    # 将稀疏矩阵转为普通矩阵
    housing_cat_1hot.toarray()
```

再次使用编码器的categories_实例变量来得到类别列表：

```python
    cat_encoder.categories_
```

### 2.5.3 自定义转换器

虽然Scikit-Learn提供了许多有用的转换器，但是仍需要为一些诸如自定义清理操作或组合特定属性等任务编写自己的转换器。由于Scikit-Learn依赖于鸭子类型的编译，而不是继承，所以目前所需要的只是创建一个类，然后应用以下三种方法：fit()（返回self）、transform()、fit_transform()。

可以通过**添加TransformerMixin作为基类**，直接得到最后一种方法。同时，如果添
加BaseEstimator作为基类（并在构造函数中避免* args和** kargs），还能额外获得两种非常有用的**自动调整超参数的方法**（get_params()和set_params()）。例如，前面讨论过的组合属性，这里有个简单的转换器类，用来添加组合后的属性：

```python
    from sklearn.base import BaseEstimator, TransformerMixin

    rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room = True) : # no *atgs or **kargs
            self.add_bedrooms_per_room=add_bedrooms_per_room
        def fit(self, X, y=None):
            return self # nothing else to do
        def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:,households_ix]
            population_per_household = X[:, population_ix] / X[:,households_ix]
            if self.add_bedrooms_per_room :
                bedrooms_per_room = X[:, bedrooms_ix] / X[:,rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)
```

在本例中，转换器有一个超参数add_bedrooms_per_room默认设置为True（提供合理的默认值通常是很有帮助的）。这个超参数可以让你轻松知晓添加这个属性是否有助于机器学习算法。更一般地，如果你对数据准备的步骤没有充分的信心，就可以添加这个超参数来进行把关。这些数据准备步骤的执行越自动化，你自动尝试的组合也就越多，从而有更大可能从中找到一个重要的组合（还节省了大量时间）。

### 2.5.4 特征缩放

最重要也最需要应用到数据上的转换就是**特征缩放**。如果输入的数值属性具有非常大的比例差异，往往会导致机器学习算法的性能表现不佳，当然也有极少数特例。

同比例缩放所有属性的两种常用方法是**最小-最大缩放(min-max scaling)**和**标准化(standardization)**。

最小-最大缩放（又叫作归一化）很简单：将值重新缩放使其最终范围归于0～1之间。实现方法是将**值减去最小值并除以最大值和最小值的差**。对此，Scikit-Learn提供了一个名为MinMaxScaler的转换器。如果希望手动控制范围，那么可以通过调整超参数feature_range进行更改。

标准化则完全不一样：首先减去平均值（所以标准化值的均值总是零），然后除以方差，从而使得**结果的分布具备单位方差**。标准化不会将值限定在特定范围，且受异常值影响更小。Scikit-Learn提供了一个标准化的转换器StandadScaler。

重要的是，跟所有转换一样，缩放器**仅用来拟合训练集，而不是完整的数据集（包括测试集）**。只有这样，才能使用它们来转换训练集和测试集（和新数据）

### 2.5.5 转换流水线

许多数据转换的步骤需要以正确的顺序来执行。而Scikit-Learn正好提供了Pipeline类来支持这样的转换。下面是一个数值属性的流水线示例：

```python
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)
```

Pipeline构造函数会通过一系列名称/估算器的**配对来定义步骤序列**。除了最后一个是
估算器之外，前面都必须是转换器（也就是说，必须有fit_transform()方法）。

当调用流水线的fit()方法时，会在所有转换器上按照顺序依次调用fit_transform()，将一个调用的输出作为参数传递给下一个调用方法，直到传递到最终的估算器，则只会调用fit()方法。

流水线的方法与最终的估算器的方法相同。在本例中，最后一个估算器是StandardScaler，这是一个转换器，因此流水线有一个transform()方法，可以按顺序将所有的转换应用到数据中（这也是我们用过的fit_transform()方法）。

到目前为止，我们分别处理了类别列和数值列。拥有一个能够处理所有列的转换器会更方便，将适当的转换应用于每个列。在0.20版中，Scikit-Learn为此引入了ColumnTransformer，好消息是它与pandas DataFrames一起使用时效果很好。让我们用它来将所有转换应用到房屋数据：

```python
    from sklearn.compose import ColumnTransformer

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
```

首先导入ColumnTransformer类，接下来获得数值列名称列表和类别列名称列表，然后构造一个ColumnTransformer。构造函数需要一个元组列表，其中每个元组都包含一个名字、一个转换器，以及一个该转换器能够应用的列名字（或索引）的列表。

在此示例中，我们指定数值列使用之前定义的num_pipeline进行转换，类别列使用OneHotEncoder进行转换。最后，我们将ColumnTransformer应用到房屋数据：它将每个转换器应用于适当的列，并沿第二个轴合并输出（转换器必须返回相同数量的行）。

OneHotEncoder返回一个稀疏矩阵，而num_pipeline返回一个密集矩阵。当稀疏矩阵和密集矩阵混合在一起时，ColumnTransformer会估算最终矩阵的密度（即单元格的非零比率），如果密度低于给定的阈值，则返回一个稀疏矩阵（通过默认值为sparse_threshold=0.3）。

## 2.6 选择和训练模型

### 2.6.1 训练和评估训练集

首先,训练一个线性回归模型：

```python
    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
```

现在拥有一个可以工作的线性回归模型，用几个训练集的实例试试

```python
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    print("Prediction:", lin_reg.predict(some_data_prepared))
```

使用Scikit-Learn的mean_squared_error()函数来测量整个训练集上回归模型的RMSE：

```python
    from sklearn.metrics import mean_squared_error
    hosing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, hosing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_rmse
```

这虽然比什么都没有要好，但显然也不是一个好看的成绩：大多数区域的median_housing_values分布在120000～265000美元之间，所以典型的预测误差达到68628美元只能算是差强人意。这就是一个典型的模型对训练数据欠拟合的案例。这种情况发生时，通常意味着这些特征可能无法提供足够的信息来做出更好的预测，或者是模型本身不够强大。**想要修正欠拟合，可以通过选择更强大的模型，或为算法训练提供更好的特征，又或者减少对模型的限制等方法**。

暂时通过训练更复杂的模型来解决——训练一个DecisioTreeRegressor。这是一个非常强大的模型，它能够从数据中找到复杂的非线性关系。

```python
    from sklearn.tree import DecisionTreeRegressor

    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)

    hosing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, hosing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_rmse
```

结果显示预测误差为0。有可能是模型完美解决了该问题，但更有可能的是模型对数据过拟合。

### 2.6.2 使用交叉验证来更好地进行评估

评估决策树模型的一种方法是使用train_test_split函数将训练集分为较小的训练集和验
证集，然后根据这些较小的训练集来训练模型，并对其进行评估。

另一个不错的选择是使用Scikit-Learn的K-折交叉验证功能。它将训练集随机分割成10个不同的子集，每个子集称为一个折叠，然后对决策树模型进行10次训练和评估——每次挑选1个折叠进行评估，使用另外的9个折叠进行训练。产生的结果是一个包含10次评估分数的数组：

```python
    # K-折交叉验证功能
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
```

Scikit-Learn的交叉验证功能更倾向于使用效用函数（越大越好）而不是成本函数（越小越好），所以计算分数的函数实际上是负的MSE（一个负值）函数，这就是为什么上面的代码在计算平方根之前会先计算出-scores。

```python
    def display_scores(scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())

    display_scores(tree_rmse_scores)

    Scores: [69327.01708558 65486.39211857 71358.25563341 69091.37509104
 70570.20267046 75529.94622521 69895.20650652 70660.14247357
 75843.74719231 68905.17669382]
Mean: 70666.74616904806
Standard deviation: 2928.322738055112
```

这次的决策树模型好像不如之前表现得好。事实上，它看起来简直比线性回归模型还要糟糕！请注意，**交叉验证不仅可以得到一个模型性能的评估值，还可以衡量该评估的精确度（即其标准差）**。这里该决策树得出的评分约为70666，上下浮动±2928。如果只使用了一个验证集，就收不到这样的结果信息。交叉验证的代价就是要多次训练模型，因此也不是永远都行得通。

同样计算一下线性回归模型的评分：

```python
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)

    Scores: [66782.73843989 66960.118071   70347.95244419 74739.57052552
 68031.13388938 71193.84183426 64969.63056405 68281.61137997
 71552.91566558 67665.10082067]
Mean: 69052.46136345083
Standard deviation: 2731.674001798351
```

决策树模型确实是严重过拟合了，以至于表现得比线性回归模型还要糟糕。

再来尝试一种模型：RandomForestRegressor（随机森林），**通过对特征的随机子集进行许多个决策树的训练，然后对其预测取平均**。在多个模型的基础之上建立模型，称为**集成学习**，这是进一步推动机器学习算法的好方法：

```python
    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    display_scores(forest_rmse_scores)

    Scores: [49469.54113454 47669.08053493 49591.14597627 52145.2679584
 49565.22756664 53351.05913562 48786.96533368 47817.30125209
 53097.80169069 49907.17425048]
Mean: 50140.056483335524
Standard deviation: 1937.7046591544113
```

这个就好多了：随机森林看起来很有戏。但是，请注意，训练集上的分数仍然远低于验证集，这意味着该模型仍然对训练集过拟合。

每一个尝试过的模型你都应该妥善保存，以便将来可以轻松回到你想要的模型当中。记得还要同时保存超参数和训练过的参数，以及交叉验证的评分和实际预测的结果。

## 2.7 微调模型

在具有有效模型的候选列表的基础上，对模型进行微调。

### 2.7.1 网格搜索
