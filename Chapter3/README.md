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

![图01_MNIST数据集](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter3/%E5%9B%BE01_MNIST%E6%95%B0%E6%8D%AE%E9%9B%86.jpg)

准备测试集。事实上，MNIST数据集已经分成训练集（前6万张图片）和测试集（最后1万张图片）了：

```python
    X_train, X_test, Y_train, Y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

同样，我们先将训练集数据混洗，这样能保证交叉验证时所有的折叠都差不多（你肯定不希望某个折叠丢失一些数字）。

## 3.2 训练二元分类器

现在先简化问题，只尝试识别一个数字，比如数字5。那么这个“数字5检测器”就是一个**二元分类器**的示例，它只能区分两个类别：5和非5。先为此分类任务创建目标向量：

```python
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
```

接着挑选一个分类器并开始训练。一个好的初始选择是**随机梯度下降（SGD）分类器**，使用Scikit-Learn的SGDClassifier类即可。这个分类器的优势是**能够有效处理非常大型的数据集**。这部分是因为SGD独立处理训练实例，一次一个（这也使得SGD非常适合在线学习），稍后我们将会看到。此时先创建一个SGDClassifier并在整个训练集上进行训练：

```python
    from sklearn.linear_model import SGDClassifier

    sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3, random_state = 42)
    sgd_clf.fit(X_train, y_train_5)
```

SGDClassifier在训练时是完全随机的（因此得名“随机”），如果你希望得到可复现的结果，需要设置参数random_state。

用它来检测数字5的图片：

```python
    sgd_clf.predict([some_digit])
```

分类器猜这个图像代表5（True）。看起来这次它猜对了！再评估一下这个模型的性能。

## 3.3 性能测量

评估分类器比评估回归器要困难得多，因此本章将用很多篇幅来讨论这个主题，同时会涉及许多性能考核的方法。

### 3.3.1 使用交叉验证测量准确率

实现交叉验证：相比于Scikit-Learn提供`cross_val_score()`这一类交叉验证的函数，有时可能希望自己能控制得多一些。在这种情况下，可以自行实现交叉验证，操作也简单明了。下面这段代码与前面的`cross_val_score()`大致相同，并打印出相同的结果。

```python
    # 自行实现交叉验证
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train_5[train_index]

        X_test_fold = X_train[test_index]
        y_test_fold = y_train_5[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred)) # prints 0.9669 0.91625 0.96785
```

每个折叠由StratifiedKFold执行分层抽样（参见第2章）产生，其所包含的各个类的比例符合整体比例。每个迭代会创建一个分类器的副本，用训练集对这个副本进行训练，然后用测试集进行预测。最后计算正确预测的次数，输出正确预测的比率。

现在，用`cross_val_score()`函数来评估SGDClassifier模型，采用K-折交叉验证法（3个折叠）。记住，K-折交叉验证的意思是将训练集分解成K个折叠（在本例中，为3折），然后每次留其中1个折叠进行预测，剩余的折叠用来训练（参见第2章）：

```python
    from sklearn.model_selection import cross_val_score
    cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
    >>>array([0.95035, 0.96035, 0.9604 ])
```

所有折叠交叉验证的准确率（正确预测的比率）超过93%？看起来挺神奇的，是吗？不过在你开始激动之前，我们来看一个蠢笨的分类器，它将每张图都分类成“非5”：

```python
    # 分类为非5的分类器
    from sklearn.base import BaseEstimator

    class Never5Classifier(BaseEstimator):
        def fit(self, X, y=None):
            return self
        
        def predict(self, X):
            return np.zeros((len(X), 1), dtype=bool)

    never_5_clf = Never5Classifier()
    cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    >>>array([0.91125, 0.90855, 0.90915])
```

没错，准确率超过90%！这是因为只有大约10%的图片是数字5，所以如果你猜一张图不是5，90%的概率你都是正确的，简直超越了大预言家！

这说明**准确率通常无法成为分类器的首要性能指标**，特别是当你处理有偏数据集时（即某些类比其他类更为频繁）

### 3.3.2 混淆矩阵

评估分类器性能的更好方法是**混淆矩阵**，其总体思路就是统计A类别实例被分成为B类别的次数。例如，要想知道分类器将数字3和数字5混淆多少次，只需要通过混淆矩阵的第5行第3列来查看。

**要计算混淆矩阵，需要先有一组预测才能将其与实际目标进行比较**。当然，可以通过测试集来进行预测，但是现在先不要动它（测试集最好留到项目的最后，准备启动分类器时再使用）。作为替代，可以使用`cross_val_predict()`函数：

```python
    # 计算混淆矩阵，使用cross_val_predict()函数：
    from sklearn.model_selection import cross_val_predict
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

与`cross_val_score()`函数一样，`cross_val_predict()`函数同样执行K-折交叉验证，但返回的不是评估分数，而是每个折叠的预测。这意味着对于每个实例都可以得到一个干净的预测（“干净”的意思是**模型预测时使用的数据在其训练期间从未见过**）。

现在可以使用`confusion_matrix()`函数来获取混淆矩阵了。只需要给出目标类别（y_train_5）和预测类别（y_train_pred）即可：

```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_train_5, y_train_pred)
    >>>array([[53892,   687],
       [ 1891,  3530]], dtype=int64)
```

混淆矩阵中的**行表示实际类别，列表示预测类别**。本例中第一行表示所有“非5”（负类）的图片中：53892张被正确地分为“非5”类别（真负类），687张被错误地分类成了“5”（假正类）；第二行表示所有“5”（正类）的图片中：1891张被错误地分为“非5”类别（假负类），3530张被正确地分在了“5”这一类别（真正类）。**一个完美的分类器只有真正类和真负类**，所以它的混淆矩阵只会在其对角线（左上到右下）上有非零值：

```python
    y_train_perfect_predictions = y_train_5
    confusion_matrix(y_train_5, y_train_perfect_predictions)
    >>>array([[54579,     0],
       [    0,  5421]], dtype=int64)
```

混淆矩阵能提供大量信息，但有时你可能希望指标更简洁一些。**正类预测的准确率**是一个有意思的指标，它也称为分类器的**精度**：

![图02_精度](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter3/%E5%9B%BE02_%E7%B2%BE%E5%BA%A6.png)

TP是真正类的数量，FP是假正类的数量。

做一个单独的正类预测，并确保它是正确的，就可以得到完美精度（精度=1/1=100%）。但这没什么意义，因为分类器会忽略这个正类实例之外的所有内容。因此，精度通常与另一个指标一起使用，这个指标就是**召回率**，也称为**灵敏度或者真正类率**：**它是分类器正确检测到的正类实例的比率**：

![图03_召回率](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter3/%E5%9B%BE03_%E5%8F%AC%E5%9B%9E%E7%8E%87.png)

FN是假负类的数量。

混淆矩阵显示了真负（左上）、假正（右上）、假负（左下）和真正（右下）的示例：

![图04_混淆矩阵示意图](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter3/%E5%9B%BE04_%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5%E7%A4%BA%E6%84%8F%E5%9B%BE.jpg)

### 3.3.3 精度和召回率

Scikit-Learn提供了计算多种分类器指标的函数，包括精度和召回率：

```python
    from sklearn.metrics import precision_score, recall_score
    precision_score(y_train_5, y_train_pred)
    recall_score(y_train_5, y_train_pred)
```

现在再看，这个5-检测器看起来并不像它的准确率那么光鲜亮眼了。当它说一张图片是5时，只有72.9%的概率是准确的，并且也只有75.6%的数字5被它检测出来了。

因此我们可以很方便地将精度和召回率组合成一个单一的指标，称为**F1分数**。当你需要一个简单的方法来比较两种分类器时，这是个非常不错的指标。**F1分数是精度和召回率的谐波平均值**（见公式3-3）。**正常的平均值平等对待所有的值，而谐波平均值会给予低值更高的权重。因此，只有当召回率和精度都很高时，分类器才能得到较高的F1分数**。

![图05_F1公式](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter3/%E5%9B%BE05_F1%E5%85%AC%E5%BC%8F.png)

要计算F1分数，只需要调用f1_score()即可：

```python
    # 计算f1
    from sklearn.metrics import f1_score
    f1_score(y_train_5, y_train_pred)
```

F1分数对那些具有相近的精度和召回率的分类器更为有利。这不一定能一直符合你的期望：在某些情况下，你更关心的是精度，而另一些情况下，你可能真正关心的是召回率。例如，假设你训练一个分类器来检测儿童可以放心观看的视频，那么你可能更青睐那种拦截了很多好视频（低召回率），但是保留下来的视频都是安全（高精度）的分类器，而不是召回率虽高，但是在产品中可能会出现一些非常糟糕的视频的分类器（这种情况下，你甚至可能会添加一个人工流水线来检查分类器选出来的视频）。反过来说，如果你训练一个分类器通过图像监控来检测小偷：你大概可以接受精度只有30%，但召回率能达到99%（当然，安保人员会收到一些错误的警报，但是几乎所有的窃贼都在劫难逃）。

遗憾的是，鱼和熊掌不可兼得，你不能同时增加精度又减少召回率，反之亦然。这称为**精度/召回率权衡**。

### 3.3.4 精度/召回率权衡

要理解这个权衡过程，我们来看看SGDClassifier如何进行分类决策。对于每个实例，它会基于决策函数计算出一个分值，如果该值大于阈值，则将该实例判为正类，否则便将其判为负类。下图显示了从左边最低分到右边最高分的几个数字。假设决策阈值位于中间箭头位置（两个5之间）：在阈值的右侧可以找到4个真正类（真的5）和一个假正类（实际上是6）。因此，在该阈值下，精度为80%（4/5）。但是在6个真正的5中，分类器仅检测到了4个，所以召回率为67%（4/6）。现在，如果提高阈值（将其挪动到右边箭头的位置），假正类（数字6）变成了真负类，因此精度得到提升（本例中提升到100%），但是一个真正类变成一个假负类，召回率降低至50%。反之，降低阈值则会在增加召回率的同时降低精度。

Scikit-Learn不允许直接设置阈值，但是可以访问它用于预测的决策分数。不是调用分类器的`predict()`方法，而是调用`decision_function()`方法，这种方法**返回每个实例的分数，然后就可以根据这些分数，使用任意阈值进行预测**了：

![图06_分类器评分](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter3/%E5%9B%BE06_%E5%88%86%E7%B1%BB%E5%99%A8%E8%AF%84%E5%88%86.jpg)

```python
    y_scores = sgd_clf.decision_function([some_digit])
    y_scores
```

```python
    threshold = 0
    y_some_digit_pred = (y_scores > threshold)
    y_some_digit_pred
    >>> array([true])
```

SGDClassifier分类器使用的阈值是0，所以前面代码的返回结果与predict()方法一样（也就是True）。我们来试试提升阈值：

```python
    threshold = 8000
    y_some_digit_pred = (y_scores > threshold)
    y_some_digit_pred
    >>> array([false])
```

这证明了提高阈值确实可以降低召回率。这张图确实是5，当阈值为0时，分类器可以检测到该图，但是当阈值提高到8000时，就错过了这张图。

那么要如何决定使用什么阈值呢？首先，使用`cross_val_predict()`函数获取训练集中所有实例的分数，但是这次需要它返回的是决策分数而不是预测结果：

```python
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
```

有了这些分数，可以使用`precision_recall_curve()`函数来计算所有可能的阈值的精度和召回率：

```python
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

最后，使用Matplotlib绘制精度和召回率相对于阈值的函数图:

```python
    def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
        plt.legend(loc="center right", fontsize=16)
        plt.xlabel("Threshold", fontsize=16)        # Not shown
        plt.grid(True)                              # Not shown
        plt.axis([-50000, 50000, 0, 1])             # Not shown



    recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
    threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]


    plt.figure(figsize=(8, 4)) 
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")
    plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")
    plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")# Not shown
    plt.plot([threshold_90_precision], [0.9], "ro")
    plt.plot([threshold_90_precision], [recall_90_precision], "ro")
    save_fig("precision_recall_vs_threshold_plot")
    plt.show()
```

![图07_精度和召回率与决策阈值](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter3/%E5%9B%BE07_%E7%B2%BE%E5%BA%A6%E5%92%8C%E5%8F%AC%E5%9B%9E%E7%8E%87%E4%B8%8E%E5%86%B3%E7%AD%96%E9%98%88%E5%80%BC.jpg)

上图中精度曲线比召回率曲线要崎岖一些的原因在于：当提高阈值时，精度有时也有可能会下降（尽管总体趋势是上升的）。另一方面，当阈值上升时，召回率只会下降，这就解释了为什么召回率的曲线看起来很平滑。

另一种找到好的精度/召回率权衡的方法是**直接绘制精度和召回率的函数图**，如图8所示（突出显示与前面相同的阈值）。

从图中可以看到，从80%的召回率往右，精度开始急剧下降。为此，可能会尽量在这个陡降之前选择一个精度/召回率权衡——比如召回率60%。当然，如何选择取决于项目的实际需要。

![图08_精度与召回率](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter3/%E5%9B%BE08_%E7%B2%BE%E5%BA%A6%E4%B8%8E%E5%8F%AC%E5%9B%9E%E7%8E%87.jpg)

假设决定将精度设为90%。查找图7并发现需要设置8000的阈值。更精确地说，你可以搜索到能提供至少90%精度的最低阈值（`np.argmax()`会给你最大值的第一个索引，在这种情况下，它表示第一个True值）：

```python
    threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
    threshold_90_precision
```

要进行预测（现在是在训练集上），除了调用分类器的`predict()`方法，也可以运行这段代码：

```python
    y_train_pred_90 = (y_scores >= threshold_90_precision)
```

检查一下这些预测结果的精度和召回率：

```python
    precision_score(y_train_5, y_train_pred_90)
    recall_score(y_train_5, y_train_pred_90)
```

现在你有一个90%精度的分类器了（或者足够接近）！如你所见，创建任意一个你想要的精度的分类器是相当容易的事情：只要阈值足够高即可！然而，如果召回率太低，精度再高，其实也不怎么有用！

### 3.3.5 ROC曲线

还有一种经常与二元分类器一起使用的工具，叫作**受试者工作特征曲线**（简称ROC）。它与精度/召回率曲线非常相似，但绘制的不是精度和召回率，而是**真正类率**（召回率的另一名称）和**假正类率**（FPR）。**FPR是被错误分为正类的负类实例比率**。它等于1减去真负类率（TNR），后者是被正确分类为负类的负类实例比率，也称为特异度。因此，**ROC曲线绘制的是灵敏度（召回率）和（1-特异度）的关系**。

要绘制ROC曲线，首先需要使用`roc_curve()`函数计算多种阈值的TPR和FPR：

```python
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```

然后，使用Matplotlib绘制FPR对TPR的曲线。下面的代码可以绘制出图9的曲线：

```python
    def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], "k--")
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate(Fall-Out)', fontsize=16)
        plt.ylabel('True Positive Rate(Recall)', fontsize=16)
        plt.grid(True)

    plt.figure(figsize=(8, 6))
    plot_roc_curve(fpr, tpr)
    fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]
    plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:") 
    plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
    plt.plot([fpr_90], [recall_90_precision], "ro")
    save_fig("roc_curve_plot")
    plt.show()
```

![图09_ROC曲线](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter3/%E5%9B%BE09_ROC%E6%9B%B2%E7%BA%BF.jpg)

同样这里再次面临一个折中权衡：召回率（TPR）越高，分类器产生的假正类（FPR）就越多。虚线表示纯随机分类器的ROC曲线、一个优秀的分类器应该离这条线越远越好（向左上角）。

有一种比较分类器的方法是测量曲线下面积（AUC）。完美的分类器的ROC AUC等于1，而纯随机分类器的ROC AUC等于0.5。Scikit-Learn提供计算ROC AUC的函数：

```python
    from sklearn.metrics import roc_auc_score
    roc_auc_score(y_train_5, y_scores)
```

由于ROC曲线与精度/召回率（PR）曲线非常相似，因此你可能会问如何决定使用哪种曲线。有一个经验法则是，**当正类非常少见或者你更关注假正类而不是假负类时，应该选择PR曲线，反之则是ROC曲线**。

训练一个`RandomForestClassifier`分类器，并比较它和`SGDClassifier`分类器的ROC曲线和ROC AUC分数。首先，获取训练集中每个实例的分数。但是由于它的工作方式不同（参见第7章），`RandomForestClassifie`r类没有`decision_function()`方法，相反，它有`dict_proba()`方法。Scikit-Learn的分类器通常都会有这两种方法中的一种（或两种都有）。`dict_proba()`方法会返回一个数组，其中每行代表一个实例，每列代表一个类别，意思是某个给定实例属于某个给定类别的概率（例如，这张图片有70%的可能是数字5）

```python
    from sklearn.ensemble import RandomForestClassifier

    forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
```

`roc_curve()`函数需要标签和分数，但是我们不提供分数，而是提供类概率。我们直接使用正类的概率作为分数值。

```python
    y_scores_forest = y_probas_forest[:, 1]
    fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, y_scores_forest)
```

现在可以绘制ROC曲线了。绘制第一条ROC曲线来看看对比结果：

```python
    recall_for_forest = tpr_forest[np.argmax(fpr_forest >= fpr_90)]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
    plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")
    plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
    plt.plot([fpr_90], [recall_90_precision], "ro")
    plt.plot([fpr_90, fpr_90], [0., recall_for_forest], "r:")
    plt.plot([fpr_90], [recall_for_forest], "ro")
    plt.grid(True)
    plt.legend(loc="lower right", fontsize=16)
    save_fig("roc_curve_comparison_plot")
    plt.show()
```

![图10_比较ROC曲线](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter3/%E5%9B%BE10_%E6%AF%94%E8%BE%83ROC%E6%9B%B2%E7%BA%BF.jpg)

`RandomForestClassifier`的ROC曲线看起来比`SGDClassifier`好很多，它离左上角更接近，因此它的ROC AUC分数也高得多：

```python
    roc_auc_score(y_train_5, y_scores_forest)
```

## 3.4 多类分类器
