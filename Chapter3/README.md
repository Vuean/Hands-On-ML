# 第三章 分类

第1章提到，最常见的有监督学习任务包括回归任务（预测值）和分类任务（预测类）。第2章探讨了一个回归任务——预测住房价格，用到了线性回归、决策树以及随机森林等各种算法（在后续章节中进一步讲解这些算法）。本章主要介绍分类系统。

## 3.1 MNIST

MNIST数据集(Mixed National Institute of Standards and Technology database)是美国国家标准与技术研究院收集整理的大型手写数字数据库，包含60000个示例的训练集以及10000个示例的测试集。

本章将使用MNIST数据集，这是一组由美国高中生和人口调查局员工手写的70000个数字的图片。每张图片都用其代表的数字标记。这个数据集被广为使用，因此也被称作是机器学习领域的“Hello World”：但凡有人想到了一个新的分类算法，都会想看看在MNIST上的执行结果。因此只要是学习机器学习的人，早晚都要面对MNIST。

获取MNIST数据集：

```python
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    mnist.keys()
```

Scikit-Learn加载的数据集通常具有类似的字典结构，包括：

- DESCR键，描述数据集。
- data键，包含一个数组，每个实例为一行，每个特征为一列。
- target键，包含一个带有标记的数组。

例如：

```python
    # 查看数组数据
    X, y = mnist["data"], mnist["target"]
    X.shape
    >>>(70000, 784)
    y.shape
    >>>(70000,)
```

共有7万张图片，每张图片有784个特征，因为图片是28×28像素，每个特征代表了一个像素点的强度，从0（白色）到255（黑色）。先来看看数据集中的一个数字，你只需要随手抓取一个实例的特征向量，将其重新形成一个28×28数组，然后使用Matplotlib的imshow()函数将其显示出来：

```python
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    some_digit = X[0]
    some_digit_image = some_digit.reshape(28,28)

    plt.imshow(some_digit_image, cmap="binary")
    plt.axis("off")

    # saving figure
    save_fig("fig1_some_digit_plot")

    plt.show()

    # 验证结果
    y[0]
```

看起来像5，而标签告诉我们没错。

注意标签是字符，大部分机器学习算法希望是数字，让我们把y转换成整数：

```python
    y = y.astype(np.unit8)
```

![图01_MNIST数据集]()

准备测试集。事实上，MNIST数据集已经分成训练集（前6万张图片）和测试集（最后1万张图片）了：

```python
    X_train, X_test, Y_train, Y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

同样，我们先将训练集数据混洗，这样能保证交叉验证时所有的折叠都差不多（你肯定不希望某个折叠丢失一些数字）。

## 3.2 训练二元分类器