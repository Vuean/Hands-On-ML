# 第十二章 使用TensorFlow自定义模型和训练

到目前为止，仅仅使用了TensorFlow的高层API tf.keras，但它已经使我们走得很远了：我们构建了各种神经网络架构，包括回归和分类网络宽深网络以及自归一化网络，使用了各种技术，例如批归一化、dropout和学习率调度。实际上，你遇到的情况中有95％不需要tf.keras（和tf.data，见第13章）以外的任何内容。但是现在是时候深研究TensorFlow并了解其底层Python API了。当你需要额外的控制来编写自定义损失函数、自定义指标、层、模型、初始化程序、正则化函数、权重约束等时，这将非常有用。你甚至可能需要完全控制训练循环本身，例如对梯度使用特殊的变换或约束（不仅仅对它们进行裁剪），或者对网络的不同部分使用多个优化器。我们将在本章介绍所有这些情况，还将探讨如何使用TensorFlow的自动图形生成功能来增强自定义模型和训练算法。但是首先，让我们快速浏览一下TensorFlow。

## 12.1 TensorFlow快速浏览

TensorFlow是一个强大的用于数值计算的库，特别适合大规模机器学习并对其进行了微调（可以将其用于需要大量计算的任何其他操作）。它由Google Brain团队开发，并为许多Google的大规模服务提供了支持。于2015年11月开源，现在是最受欢迎的深度学习库（就论文引用、公司采用率、GitHub上的星星数量等而言）。无数项目将TensorFlow用于各种机器学习任务，例如图像分类、自然语言处理、推荐系统和时间序列预测。

TensorFlow核心功能如下：

- 核心与NumPy非常相似，但具有GPU支持。

- 支持分布式计算（跨多个设备和服务器）。

- 包含一种即时（JIT）编译器，可使其针对速度和内存使用情况来优化计算。工作方式是从Python函数中提取计算图，然后进行优化（通过修剪未使用的节点），最后有效地运行它（通过自动并行运行相互独立的操作）。

- 计算图可以导出为可移植格式，可以在一个环境中（例如在Linux上使用Python）训练TensorFlow模型，然后在另一个环境中（例如在Android设备上使用Java）运行TensorFlow模型。

- 实现了自动微分（autodiff）（见第10章和附录D），并提供了一些优秀的优化器，例如RMSProp和Nadam（见第11章），可以轻松地最小化各种损失函数。

TensorFlow在上述心功能的基础上还提供了更多其他功能：最重要的当然是tf.keras，同时还具有数据加载和预处理操作（tf.data、tf.io等）、图像处理操作（tf.image）、信号处理操作（tf.signal）等（有关TensorFlow的Python API的概述见图1）。

![fig01_TensorFlow的Python API](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter12/figures/fig01_TensorFlow%E7%9A%84Python%20API.jpg)

在最底层，每个TensorFlow操作（以下简称op）都是使用高效的C++代码实现的。许多操作都有称为内核的多种实现：每个内核专用于特定的设备类型，例如CPU、GPU甚至TPU（张量处理单元）。如你所知，GPU可以通过将GPU分成许多较小的块并在多个GPU线程中并行运行它们来极大地加快计算速度。TPU甚至更快：它们是专门为深度学习操作而构建的定制ASIC芯片（我们将在第19章中讨论如何利用GPU或TPU来使用
TensorFlow）。

TensorFlow的架构如图2所示。大多数时候，代码使用高级API（尤其是tf.keras和tf.data）。但是当需要更大的灵活性时，可以使用较低级别的Python API直接处理张量。注意也可以使用其他语言的API。无论如何，TensorFlow的执行引擎都会有效地运行操作，如果你告诉它，它也可以跨多个设备和机器运行。

![fig02_TensorFlow的架构](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter12/figures/fig02_TensorFlow%E7%9A%84%E6%9E%B6%E6%9E%84.jpg)

TensorFlow不仅可以在Windows、Linux和macOS上运行，而且可以在移动设备（使用TensorFlow Lite）上运行，包括iOS和Android（见第19章）。如果不想使用PythonAPI，则可以使用C++、Java、Go和Swift API。甚至还有一个名为TensorFlow.js的JavaScript实现，可以直接在浏览器中运行模型。

TensorFlow不仅仅是函数库，更是广泛的生态系统的核心。首先，有TensorBoard可以进行可视化（见第10章）。接下来是TensorFlow Extended（TFX），它是Google为TensorFlow项目进行生产环境而构建的一组库：它包括用于数据验证、预处理、模型分析和服务的工具（使用TF Serving，见第19章）。Google的TensorFlow Hub提供了一种轻松下载和重用预训练的神经网络的方法。你还可以在TensorFlow的模型花园中获得许多神经网络架构，其中一些已经过预先训练。查看TensorFlow资源和https://github.com/jtoy/awesome-tensorflow，了解更多基于TensorFlow的项目。可以在GitHub上找到数百个TensorFlow项目，通常很容易找到你想要的代码。

最后但并非不重要的一点是，TensorFlow拥有一支热情而乐于助人的开发人员组成的团队以及一个大型社区，致力于对其进行改进。要问技术问题，你应该使用http://stackoverflow.com/并用tensorflow和python标记你的问题。你可以通过GitHub提交错误和功能请求。有关一般讨论，请加入Google讨论组。


## 12.2 像NumPy一样使用TensorFlow

TensorFlow的API一切都围绕张量，张量从一个操作流向另一个操作，因此命名为TensorFlow。张量非常类似NumPy的ndarray，它通常是一个**多维度组**，但它也可以保存标量（简单值，例如42）。当我们创建自定义成本函数、自定义指标、自定义层等时，这些张量将非常重要，因此让我们来看看如何创建和操作它们。

### 12.2.1 张量和操作

可以使用`tf.constant()`创建张量。例如，这是一个张量，表示具有两行三列浮点数的矩阵：

```python
    tf.constant([[1., 2., 3.], [4., 5., 6.]]) # matrix
    >>> <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[1., 2., 3.],
        [4., 5., 6.]], dtype=float32)>

    tf.constant(42) # scalar
    >>> <tf.Tensor: shape=(), dtype=int32, numpy=42>
```

就像ndarray一样，tf.Tensor具有形状和数据类型（dtype）：

```python
    t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    t
    >>> <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[1., 2., 3.],
        [4., 5., 6.]], dtype=float32)>
    
    t.shape
    >>> TensorShape([2, 3])

    t.dtype
    >>> tf.float32
```

索引的工作方式非常类似于NumPy：

```python
    t[:, 1:]
    >>> <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[2., 3.],
        [5., 6.]], dtype=float32)>

    t[..., 1, tf.newaxis]
    >>> <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
    array([[2.],
        [5.]], dtype=float32)>
```

最重要的是，可以使用各种张量操作：

```python
    t + 10
    >>> <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[11., 12., 13.],
        [14., 15., 16.]], dtype=float32)>
    
    tf.square(t)
    >>> <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[ 1.,  4.,  9.],
        [16., 25., 36.]], dtype=float32)>

    t @ tf.transpose(t)
    >>> <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[14., 32.],
        [32., 77.]], dtype=float32)>
```

请注意，t+10等效于调用`tf.add(t,10)`（实际上，Python调用了方法`t.__add__(10)`，该方法仅调用`tf.add(t,10)`）。还支持其他运算符，例如-和*。@运算符是在Python 3.5中添加的，用于矩阵乘法，等效于调用`tf.matmul()`函数。

可以找到所需的所有基本数学运算（`tf.add()`、`tf.multiply()`、`tf.square()`、`tf.exp()`、`tf.sqrt()`等）以及在NumPy找到的大多数运算（例如`tf.reshape()`、`tf.squeeze()`、`tf.tile()`）。某些函数的名称与NumPy中的名称不同。例如，`tf.reduce_mean()`、`tf.reduce_sum()`、`tf.reduce_max()`和`tf.math.log()`等效于`np.mean()`、`np.sum()`、`np.max()`和`np.log()`。名称不同时，通常有充分的理由。例如，在TensorFlow中，你必须编写`tf.transpose(t)`，不能就像在NumPy中一样只是写`t.T`。原因是`tf.transpose()`函数与NumPy的T属性没有完全相同的功能：在TensorFlow中，使用自己的转置数据副本创建一个新的张量，而在NumPy中，`t.T`只是相同数据的转置视图。类似地，`tf.reduce_sum()`操作之所以这样命名，是因为其GPU内核（即GPU实现）使用的reduce算法不能保证元素添加的顺序：因为32位浮点数的精度有限，因此每次你调用此操作时，结果可能会稍有不同。`tf.reduce_mean()`也是如此（当然`tf.reduce_max()`是确定性的）。

许多函数和类都有别名。例如，`tf.add()`和`tf.math.add()`是同一函数。这使得TensorFlow可以为最常见的操作使用简洁的名称，同时保留组织良好的软件包。

#### Keras的底层API

Keras API在keras.backend中有自己的底层API。它包含诸如`square()`、`exp()`和`sqrt()`等函数。在tf.keras中，这些函数通常只调用相应的TensorFlow操作。如果要编写可移植到其他Keras实现中的代码，则应使用这些Keras函数。但是它们仅涵盖TensorFlow中所有可用函数的子集，因此在本书中，我们直接使用TensorFlow操作。这是使用keras.backend的简单示例，它通常简称为K：

```python
    from tensorflow import keras
    K = keras.backend
    K.square(K.transpose(t)) + 10
    >>> <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
    array([[11., 26.],
        [14., 35.],
        [19., 46.]], dtype=float32)>
```

### 12.2.2 张量和NumPy

张量可以与NumPy配合使用：可以用NumPy数组创建张量，反之亦然。甚至还可以将TensorFlow操作应用于NumPy数组，将NumPy操作应用于张量：

```python
    a = np.array([2., 4., 5.])
    tf.constant(a)
    >>> <tf.Tensor: shape=(3,), dtype=float64, numpy=array([2., 4., 5.])>

    t.numpy()
    >>> array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)

    np.array(t)
    >>> array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)
    
    tf.square(a)
    >>> <tf.Tensor: shape=(3,), dtype=float64, numpy=array([ 4., 16., 25.])>

    np.square(t)
    >>> array([[ 1.,  4.,  9.],
       [16., 25., 36.]], dtype=float32)
```

请注意，默认情况下NumPy使用64位精度，而TensorFlow使用32位精度。这是因为32位精度通常对于神经网络来说绰绰有余，而且运行速度更快且使用的RAM更少。因此，当从NumPy数组创建张量时，需确保设置dtype=tf.float32。

### 12.2.3 类型转换

类型转换会严重影响性能，并且自动完成转换很容易被忽视。为了避免这种情况，TensorFlow不会自动执行任何类型转换：如果对不兼容类型的张量执行操作，会引发异常。例如，不能把浮点张量和整数张量相加，甚至不能相加32位浮点和64位浮点：

```python
    try:
        tf.constant(2.0) + tf.constant(40)
    except tf.errors.InvalidArgumentError as ex:
        print(ex)
    >>> cannot compute AddV2 as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:AddV2]

    try:
        tf.constant(2.0) + tf.constant(40., dtype=tf.float64)
    except tf.errors.InvalidArgumentError as ex:
        print(ex)
    >>> cannot compute AddV2 as input #1(zero-based) was expected to be a float tensor but is a double tensor [Op:AddV2]
```

虽然该规则会有点烦人，但是这是必须的！当确实需要转换类型时，可以使用`tf.cast()`：

```python
    t2 = tf.constant(40., dtype=tf.float64)
    tf.constant(2.0) + tf.cast(t2, tf.float32)
    >>> <tf.Tensor: shape=(), dtype=float32, numpy=42.0>
```

### 12.2.4 变量

到目前为止，接触的tf.Tensor值是不变的，无法修改它们。这意味着当前还不能使用常规张量在神经网络中实现权重，因为它们需要通过反向传播进行调整。另外还可能需要随时间改变其他参数（例如动量优化器跟踪过去的梯度）。因此需要的是`tf.Variable`：

```python
    v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
    v
    >>> <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
    array([[1., 2., 3.],
        [4., 5., 6.]], dtype=float32)>
```

`tf.Variable`的行为与`tf.Tensor`的行为非常相似：可以使用它执行相同的操作，它在NumPy中也可以很好地发挥作用，并且对类型也很挑剔。但是也可以使用`assign()`方法（或`assign_add()`或`assign_sub()`，给变量增加或减少给定值）进行修改。还可以通过使用单元（或切片）的`assign()`方法（直接指定将不起作用）或使用`scatter_update()`或`scatter_nd_update()`方法来修改单个单元（或切片）：

```python
    v.assign(2 * v)
    v[0, 1].assign(42)
    v[:, 2].assign([0., 1.])
```

实际上，你几乎不需要手动创建变量，因为Keras提供了`add_weight()`方法，我们将看到该方法会为你解决这个问题。而且模型参数通常由优化器直接更新，因此你几乎不需要手动更新变量。

### 12.2.5 其他数据结构

TensorFlow支持其他几种数据结构，包括以下内容：

- 稀疏张量（tf.SparseTensor）

    有效地表示主要包含零的张量。tf.sparse程序包包含稀疏张量的操作。

- 张量数组（tf.TensorArray）

    张量的列表。默认情况下，它们的大小是固定的，但可以选择动态设置。它们包含的所有张量必须具有相同的形状和数据类型。

- 不规则张量（tf.RaggedTensor）

    表示张量列表的静态列表，其中每个张量具有相同的形状和数据类型。tf.ragged程序包包含用于不规则的张量的操作。

- 字符串张量
    tf.string类型的常规张量。它们表示字节字符串，而不是Unicode字符串，因此如果使用Unicode字符串（常规的Python 3字符串，例如"café"）创建字符串张量，则它将自动被编码为UTF-8（例如，b"caf\xc3\xa9"）。或者，你可以使用类型为tf.int32的张量来表示Unicode字符串，其中每个项都表示一个Unicode代码点（例如[99、97、102、233]）。tf.strings包（带有s）包含用于字节字符串和Unicode字符串的操作（并将它们转换为另一个）。重要的是要注意，tf.string是原子级的，这意味着它的长度不会出现在张量的形状中。一旦你将其转换为Unicode张量（即包含Unicode代码点的tf.int32类型的张量）后，长度就会显示在形状中。

- 集合

    表示为常规张量（或稀疏张量）。例如，tf.constant（[[[1，2]，[3，4]]）代表两个集合{1，2}和{3，4}。通常，每个集合由张量的最后一个轴上的向量表示。可使用tf.sets包中的操作来操作集。

- 队列

    跨多个步骤存储的张量。TensorFlow提供了各种队列：简单的先进先出（FIFO）队列（FIFOQueue），可以区分某些元素优先级的队列（PriorityQueue），将其元素（RandomShuffleQueue）随机排序，通过填充（PaddingFIFOQueue）批处理具有不同形状的元素。这些类都在tf.queue包中。

有了张量-运算-变量和各种数据结构，你现在就可以自定义模型和训练算法了！

## 12.3 定制模型和训练算法

从创建一个自定义损失函数开始，这是一个简单而常见的用例。

### 12.3.1 自定义损失函数

