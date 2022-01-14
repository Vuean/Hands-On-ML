# 第七章 集成学习和随机森林

如果你随机向几千个人询问一个复杂问题，然后汇总他们的回答。在许多情况下，你会发现，这个汇总的回答比专家的回答还要好，这被称为**群体智慧**。同样，如果你聚合一组预测器（比如分类器或回归器）的预测，得到的预测结果也比最好的单个预测器要好。这样的一组预测器称为**集成**，所以这种技术也被称为**集成学习**，而一个集成学习算法则被称为**集成方法**。

例如，你可以训练一组决策树分类器，每一棵树都基于训练集不同的随机子集进行训练。做出预测时，你只需要获得所有树各自的预测，然后给出得票最多的类别作为预测结果（见第6章练习题8）。这样一组决策树的集成被称为**随机森林**，尽管很简单，但它是迄今可用的最强大的机器学习算法之一。

此外，正如我们在第2章讨论过的，在项目快要结束时，你可能已经构建好了一些不错的预测器，这时就可以通过集成方法将它们组合成一个更强的预测器。事实上，在机器学习竞赛中获胜的解决方案通常都涉及多种集成方法（最知名的是Nerflix大奖赛）。

本章我们将探讨最流行的几种集成方法，包括`bagging`、`boosting`、`stacking`等，也将探索随机森林。

## 7.1 投票分类器

假设你已经训练好了一些分类器，每个分类器的准确率约为80%。大概包括一个逻辑回归分类器、一个SVM分类器、一个随机森林分类器、一个K-近邻分类器，或许还有更多（见图1）。

![fig01_训练多种分类器](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig01_%E8%AE%AD%E7%BB%83%E5%A4%9A%E7%A7%8D%E5%88%86%E7%B1%BB%E5%99%A8.jpg)

这时，要创建出一个更好的分类器，最简单的办法就是聚合每个分类器的预测，然后将得票最多的结果作为预测类别。这种大多数投票分类器被称为**硬投票分类器**。

![fig02_硬投票分类器预测](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig02_%E7%A1%AC%E6%8A%95%E7%A5%A8%E5%88%86%E7%B1%BB%E5%99%A8%E9%A2%84%E6%B5%8B.jpg)

你会多少有点惊讶地发现，这个投票法分类器的准确率通常比集成中最好的分类器还要高。事实上，即使每个分类器都是弱学习器（意味着它仅比随机猜测好一点），通过集成依然可以实现一个强学习器（高准确率），只要有足够大数量并且足够多种类的弱学习器即可。

这怎么可能呢？下面这个类比可以帮助你掀开这层神秘面纱。假设你有一个略微偏倚的硬币，它有51%的可能正面数字朝上，49%的可能背面花朝上。如果你掷1000次，你大致会得到差不多510次数字和490次花，所以正面是大多数。而如果你做数学题，你会发现，“在1000次投掷后，大多数为正面朝上”的概率接近75%。投掷硬币的次数越多，这个概率越高（例如，投掷10 000次后，这个概率攀升至97%）。这是因为大数定理导致的：随着你不断投掷硬币，正面朝上的比例越来越接近于正面的概率（51%）。图3显示了10条偏倚硬币的投掷结果。可以看出随着投掷次数的增加，正面的比例逐渐接近51%，最终所有10条线全都接近51%，并且始终位于50%以上。

![fig03_大数定理](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig03_%E5%A4%A7%E6%95%B0%E5%AE%9A%E7%90%86.jpg)

同样，假设你创建了一个包含1000个分类器的集成，每个分类器都只有51%的概率是正确的（几乎不比随机猜测强多少）。如果你以大多数投票的类别作为预测结果，可以期待的准确率高达75%。但是，这基于的前提是所有的分类器都是完全独立的，彼此的错误毫不相关。显然这是不可能的，因为它们都是在相同的数据上训练的，很可能会犯相同的错误，所以也会有很多次大多数投给了错误的类别，导致集成的准确率有所降低。

**当预测器尽可能互相独立时，集成方法的效果最优**。获得多种分类器的方法之一就是**使用不同的算法进行训练**。这会增加它们犯不同类型错误的机会，从而提升集成的准确率。

下面的代码用Scikit-Learn创建并训练一个投票分类器，由三种不同的分类器组成：

```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    log_clf = LogisticRegression(solver="lbfgs", random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_clf = SVC(gamma="scale", random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard')

    voting_clf.fit(X_train, y_train)
```

我们来看一下测试集上每个分类器的精度：

```python
    from sklearn.metrics import accuracy_score

    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

投票分类器略胜于所有单个分类器。

如果所有分类器都能够估算出类别的概率（即有`predict_proba()`方法），那么你可以将概率在所有单个分类器上平均，然后让Scikit-Learn给出平均概率最高的类别作为预测。这被称为软投票法。通常来说，它比硬投票法的表现更优，因为它给予那些高度自信的投票更高的权重。而所有你需要做的就是用`voting="soft"`代替`voting="hard"`，并确保所有分类器都可以估算出概率。默认情况下，SVC类是不行的，所以你需要将其超参数`probability`设置为`True`（这会导致SVC使用交叉验证来估算类别概率，减慢训练速度，并会添加`predict_proba()`方法）。如果修改上面代码为使用软投票，你会发现投票分类器的准确率达到91.2%以上。

## 7.2 bagging和pasting

前面提到，获得不同种类分类器的方法之一是使用不同的训练算法。还有另一种方法是每个预测器使用的算法相同，但是在不同的训练集随机子集上进行训练。采样时如果将样本放回，这种方法叫作bagging（bootstrap aggregating的缩写，也叫**自举汇聚法**）。采样时样本不放回，这种方法则叫作pasting。

换句话说，bagging和pasting都允许训练实例在多个预测器中被多次采样，但是**只有bagging允许训练实例被同一个预测器多次采样**。采样过程和训练过程如图4所示。

![fig04_bagging和pasting](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig04_bagging%E5%92%8Cpasting.jpg)

如图4所示，你可以通过不同的CPU内核甚至不同的服务器并行地训练预测器。类似地，预测也可以并行。这正是bagging和pasting方法如此流行的原因之一，它们非常易于扩展。

### 7.2.1 Scikit-Learn中的bagging和pasting

Scikit-Learn提供了一个简单的API，可用`BaggingClassifier`类进行bagging和pasting（或BaggingRegressor用于回归）。以下代码训练了一个包含500个决策树分类器的集成，每次从训练集中随机采样100个训练实例进行训练，然后放回（这是一个bagging的示例，如果你想使用pasting，只需要设置`bootstrap=False`即可）。参数`n_jobs`用来指示Scikit-Learn用多少CPU内核进行训练和预测（-1表示让Scikit-Learn使用所有可用内核）：

```python
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1
    )
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)
```

如果基本分类器可以估计类别概率（如果它具有`predict_proba()`方法），则`BaggingClassifier`自动执行软投票而不是硬投票，在决策树分类器中就是这种情况。

图5比较了两种决策边界，一种是单个决策树，一种是由500个决策树组成的bagging集成（来自前面的代码），二者均在卫星数据集上训练完成。可以看出，**集成预测的泛化效果很可能会比单独的决策树要好一些**：二者偏差相近，但是集成的方差更小（两边训练集上的错误数量差不多，但是集成的决策边界更规则）。

![fig05_A single Decision Tree (left) versus a bagging ensemble of 500 trees (right)](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig05_A%20single%20Decision%20Tree%20(left)%20versus%20a%20bagging%20ensemble%20of%20500%20trees%20(right).jpg)

由于自举法给每个预测器的训练子集引入了更高的多样性，所以最后bagging比pasting的偏差略高，但这也意味着预测器之间的关联度更低，所以集成的方差降低。总之，bagging生成的模型通常更好，这也就是为什么它更受欢迎。但是，如果你有充足的时间和CPU资源，可以使用交叉验证来对bagging和pasting的结果进行评估，再做出最合适的选择。

### 7.2.2 包外评估

对于任意给定的预测器，使用bagging，有些实例可能会被采样多次，而有些实例则可能根本不被采样。`BaggingClassifier`默认采样m个训练实例，然后放回样本（`bootstrap=True`），m是训练集的大小。这意味着对每个预测器来说，平均只对63%的训练实例进行采样。剩余37%未被采样的训练实例称为**包外（oob）**实例。注意，对所有预测器来说，这是不一样的37%。

由于预测器在训练过程中从未看到oob实例，因此可以在这些实例上进行评估，而无须单独的验证集。你可以通过平均每个预测器的oob评估来评估整体。

在Scikit-Learn中，创建`BaggingClassifier`时，设置`oob_score=True`就可以请求在训练结束后自动进行包外评估。下面的代码演示了这一点。通过变量`oob_score_`可以得到最终的评估分数。

```python
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        bootstrap=True, oob_score=True, random_state=40)
    bag_clf.fit(X_train, y_train)
    bag_clf.oob_score_
```

根据此oob评估，此`BaggingClassifier`能在测试集上达到约90.1%的准确率。让我们验证一下：

```python
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pred))
```

每个训练实例的包外决策函数也可以通过变量`oob_decision_function_`获得。本例中（基本预测器有`predict_proba()`方法），决策函数返回的是每个实例的类别概率。例如，包外评估估计，第二个训练实例有68.25%的概率属于正类（以及31.75%的概率属于负类）。

## 7.3 随机补丁和随机子空间

`BaggingClassifier`类也支持对特征进行采样。采样由两个超参数控制：`max_features`和`bootstrap_features`。它们的工作方式与`max_samples`和`bootstrap`相同，但用于特征采样而不是实例采样。因此，每个预测器将用输入特征的随机子集进行训练。

这对于处理高维输入（例如图像）特别有用。对训练实例和特征都进行抽样，这称为**随机补丁方法**。而保留所有训练实例（即`bootstrap=False`并且`max_samples=1.0`）但是对特征进行抽样（即`bootstrap_features=True`并且/或`max_features<1.0`），这被称为**随机子空间法**。

对特征抽样给预测器带来更大的多样性，所以以略高一点的偏差换取了更低的方差。

## 7.4 随即森林

前面已经提到，随机森林是决策树的集成，通常用`bagging`（有时也可能是`pasting`）方法训练，训练集大小通过`max_samples`来设置。除了先构建一个`BaggingClassifier`然后将其传输到`DecisionTreeClassifier`，还有一种方法就是使用`RandomForestClassifier`类，这种方法更方便，对决策树更优化（同样，对于回归任务也有一个`RandomForestRegressor`类）。以下代码使用所有可用的CPU内核，训练了一个拥有500棵树的随机森林分类器（每棵树限制为最多16个叶节点）：

```python
    from sklearn.ensemble import RandomForestClassifier

    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    rnd_clf.fit(X_train, y_train)

    y_pred_rf = rnd_clf.predict(X_test)
```

除少数例外，`RandomForestClassifier`具有`DecisionTreeClassifier`的所有超参数（以控制树的生长方式），以及`BaggingClassifier`的所有超参数来控制集成本身。

随机森林在树的生长上引入了更多的随机性：分裂节点时不再是搜索最好的特征（见第6章），而是在一个随机生成的特征子集里搜索最好的特征。这导致决策树具有更大的多样性，（再一次）用更高的偏差换取更低的方差，总之，还是产生了一个整体性能更优的模型。下面的`BaggingClassifier`大致与前面的`RandomForestClassifier`相同：

```python
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
        n_estimators=500, random_state=42)
```

### 7.4.1 极端随机树

如前所述，在随机森林里单棵树的生长过程中，每个节点在分裂时仅考虑到了一个随机子集所包含的特征。如果我们对每个特征使用随机阈值，而不是搜索得出的最佳阈值（如常规决策树），则可能让决策树生长得更加随机。

这种极端随机的决策树组成的森林称为**极端随机树集成**（或简称Extra-Trees）。同样，它也是以更高的偏差换取了更低的方差。极端随机树训练起来比常规随机森林要快很多，因为在每个节点上找到每个特征的最佳阈值是决策树生长中最耗时的任务之一。

使用Scikit-Learn的`ExtraTreesClassifier`类可以创建一个极端随机树分类器。它的API与`RandomForestClassifier`类相同。同理，`ExtraTreesRegressor`类与`RandomForestRegressor`类的API也相同。

通常来说，很难预先知道一个`RandomForestClassifier`类是否会比一个`ExtraTreesClassifier`类更好或是更差。唯一的方法是两种都尝试一遍，然后使用交叉验证（还需要使用网格搜索调整超参数）进行比较。

### 7.4.2 特征重要性

**随机森林的另一个好特性是它们使测量每个特征的相对重要性变得容易**。Scikit-Learn通过查看使用该特征的树节点平均（在森林中的所有树上）减少不纯度的程度来衡量该特征的重要性。更准确地说，它是一个加权平均值，其中每个节点的权重等于与其关联的训练样本的数量（见第6章）。

Scikit-Learn会在训练后为每个特征自动计算该分数，然后对结果进行缩放以使所有重要性的总和等于1。你可以使用`feature_importances_`变量来访问结果。例如，以下代码在鸢尾花数据集上训练了`RandomForestClassifier`（在第4章中介绍），并输出每个特征的重
要性。看起来最重要的特征是花瓣长度（44%）和宽度（42%），而花萼的长度和宽度则相对不那么重要（分别是11%和2%）：

```python
    from sklearn.datasets import load_iris
    iris = load_iris()
    rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
    rnd_clf.fit(iris["data"], iris["target"])
    for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
        print(name, score)
```

同样，如果在MNIST数据集上训练随机森林分类器（在第3章中介绍）并绘制每个像素的重要性，则会得到如图6所示的图像。

![fig06_MNIST像素的重要性](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig06_MNIST%E5%83%8F%E7%B4%A0%E7%9A%84%E9%87%8D%E8%A6%81%E6%80%A7.jpg)

随机森林非常便于你快速了解哪些特征是真正重要的，特别是在需要执行特性选择时。

## 7.5 提升法

**提升法**（boosting，最初被称为假设提升）是指可以将几个弱学习器结合成一个强学习器的任意集成方法。**大多数提升法的总体思路是循环训练预测器**，每一次都对其前序做出一些改正。可用的提升法有很多，但目前最流行的方法是**AdaBoost**（Adaptive Boosting的简称）和**梯度提升**。我们先从AdaBoost开始介绍。

### 7.5.1 AdaBoost

新预测器对其前序进行纠正的方法之一就是**更多地关注前序欠拟合的训练实例**，从而使新的预测器不断地越来越专注于难缠的问题，这就是AdaBoost使用的技术。

例如，当训练AdaBoost分类器时，该算法**首先训练一个基础分类器**（例如决策树），并使用它对训练集进行预测。然后，**该算法会增加分类错误的训练实例的相对权重**。然后，它使用更新后的权重训练第二个分类器，并再次对训练集进行预测，更新实例权重，以此类推（见图7）。

![fig07_AdaBoost循环训练](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig07_AdaBoost%E5%BE%AA%E7%8E%AF%E8%AE%AD%E7%BB%83.jpg)

图8显示了在卫星数据集上5个连续的预测器的决策边界（在本例中，每个预测器都使用RBF核函数的高度正则化的SVM分类器）。第一个分类器产生了许多错误实例，所以这些实例的权重得到提升。因此第二个分类器在这些实例上的表现有所提升，然后第三个、第四个......右图绘制的是相同预测器序列，唯一的差别在于**学习率减半**（即每次迭代仅提升一半错误分类的实例的权重）。可以看出，AdaBoost这种依序循环的学习技术跟梯度下降有一些异曲同工之处，差别只在于——不再是调整单个预测器的参数使成本函数最小化，而是不断在集成中加入预测器，使模型越来越好。

![fig08_连续预测器的决策边界](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig08_%E8%BF%9E%E7%BB%AD%E9%A2%84%E6%B5%8B%E5%99%A8%E7%9A%84%E5%86%B3%E7%AD%96%E8%BE%B9%E7%95%8C.jpg)

一旦全部预测器训练完成，集成整体做出预测时就跟`bagging`或`pasting`方法一样了，除非预测器有不同的权重，因为它们总的准确率是基于加权后的训练集。

这种依序学习技术有一个重要的缺陷就是**无法并行**（哪怕只是一部分），因为每个预测器只能在前一个预测器训练完成并评估之后才能开始训练。因此，在扩展方面，它的表现不如`bagging`和`pasting`方法。

让我们仔细看看AdaBoost算法。每个实例权重 $w^{(i)}$ 最初设置为 $1\over m$ 。对第一个预测器进行训练，并根据训练集计算其加权误差率r1，请参见公式7-1。

![fig09_公式7-1_第j个预测器的加权误差率](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig09_%E5%85%AC%E5%BC%8F7-1_%E7%AC%ACj%E4%B8%AA%E9%A2%84%E6%B5%8B%E5%99%A8%E7%9A%84%E5%8A%A0%E6%9D%83%E8%AF%AF%E5%B7%AE%E7%8E%87.jpg)

$\hat y^{(i)}_j$ 是第i个实例的第j个预测器的预测。

预测器的权重 $α_j$ 通过公式7-2来计算，其中η是学习率超参数（默认为1）。预测器的准确率越高，其权重就越高。如果它只是随机猜测，则其权重接近于零。但是，如果大部分情况下它都是错的（也就是准确率比随机猜测还低），那么它的权重为负。

![fig10_公式7-2_预测器权重](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig10_%E5%85%AC%E5%BC%8F7-2_%E9%A2%84%E6%B5%8B%E5%99%A8%E6%9D%83%E9%87%8D.jpg)

接下来，AdaBoost算法使用公式7-3更新实例权重，从而提高了误分类实例的权重。

![fig11_公式7-3_权重更新规则](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig11_%E5%85%AC%E5%BC%8F7-3_%E6%9D%83%E9%87%8D%E6%9B%B4%E6%96%B0%E8%A7%84%E5%88%99.jpg)

然后对所有实例权重进行归一化（即除以 $\sum^m_{i=1}w^{(i)}$）。

最后，使用更新后的权重训练一个新的预测器，然后重复整个过程（计算新预测器的权重，更新实例权重，然后对另一个预测器进行训练，等等）。当到达所需数量的预测器或得到完美的预测器时，算法停止。

预测的时候，AdaBoost就是简单地计算所有预测器的预测结果，并使用预测器权重 $α_j$ 对它们进行加权。最后，得到大多数加权投票的类就是预测器给出的预测类（见公式7-4）。

![fig12_公式7-4_AdaBoost预测](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig12_%E5%85%AC%E5%BC%8F7-4_AdaBoost%E9%A2%84%E6%B5%8B.jpg)

其中N是预测器的数量。

Scikit-Learn使用的其实是AdaBoost的一个多分类版本，叫作SAMME（基于多类指数损失函数的逐步添加模型）。当只有两类时，SAMME即等同于AdaBoost。此外，如果预测器可以估算类概率（即具有`redict_proba()`方法），Scikit-Learn会使用一种SAMME的变体，称为SAMME.R（R代表“Real”），它依赖的是类概率而不是类预测，通常表现更好。

下面的代码使用Scikit-Learn的`AdaBoostClassifier`（正如你猜想的，还有一个`AdaBoostRegressor`类）训练了一个AdaBoost分类器，它基于200个单层决策树。顾名思义，单层决策树就是`max_depth=1`的决策树，换言之，就是一个决策节点加两个叶节点。这是`AdaBoostClassifier`默认使用的基础估算器。

```python
    from sklearn.ensemble import AdaBoostClassifier

    ada_clf = AdaBoostClassifier(
        DecisionTreeClassfier(max_depth=1, n_estimators=200, algorithm_="SAMMe.R", learning_rate=0.5)
    )
    ada_clf.fit(X_train, y_train)
```

如果你的AdaBoost集成过度拟合训练集，你可以试试减少估算器数量，或是提高基础估算器的正则化程度。

### 7.5.2 梯度提升

另一个非常受欢迎的提升法是梯度提升。与AdaBoost一样，梯度提升也是**逐步在集成中添加预测器，每一个都对其前序做出改正**。不同之处在于，它不是像AdaBoost那样在每个迭代中调整实例权重，而是**让新的预测器针对前一个预测器的残差进行拟合**。

我们来看一个简单的回归示例，使用决策树作为基础预测器（梯度提升当然也适用于回归任务），这被称为**梯度树提升**或者是**梯度提升回归树**（GBRT）。首先，在训练集（比如带噪声的二次训练集）上拟合一个`DecisionTreeRegressor`：

```python
    from sklearn.tree import DecisionTreeRegressor

    tree_reg1 = DecisionTreeRegressor(max_depth =2, random_state =42)
    tree_reg1.fit(X, y)
```

针对第一个预测器的残差，训练第二个`DecisionTreeRegressor`：

```python
    y2 = y - tree_reg1.predict(X)
    tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg2.fit(X, y2)
```

针对第二个预测器的残差，训练第三个回归器：

```python
    y3 = y2 - tree_reg2.predict(X)

    tree_reg3 = DecisionTreeRegressor(max_depth =2, random_state =42)
    tree_reg3.fit(X, y3)
```

现在，我们有了一个包含三棵树的集成。它将所有树的预测相加，从而对新实例进行预测：

```python
    y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
```

图13的左侧表示这三棵树单独的预测，右侧表示集成的预测。第一行，集成只有一棵树，所以它的预测与第一棵树的预测完全相同。第二行是在第一棵树的残差上训练的一棵新树，从右侧可见，集成的预测等于前面两棵树的预测之和。类似地，第三行又有一棵在第二棵树的残差上训练的新树，集成的预测随着新树的添加逐渐变好。

![fig13_梯度提升示例](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig13_%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%E7%A4%BA%E4%BE%8B.jpg)

训练GBRT集成有个简单的方法，就是使用Scikit-Learn的`GradientBoostingRegressor`类。与`RandomForestRegressor`类似，它具有控制决策树生长的超参数（例如`max_depth`、`min_samples_leaf`等），以及控制集成训练的超参数，例如树的数量（`n_estimators`）。以下代码可创建上面的集成：

```python
    from sklearn.ensemble import GradientBoostingRegressor

    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
    gbrt.fit(X, y)
```

超参数`learning_rate`对每棵树的贡献进行缩放。如果你将其设置为低值，比如0.1，则需要更多的树来拟合训练集，但是预测的泛化效果通常更好，这是一种被称为**收缩的正则化技术**。图14显示了用低学习率训练的两个GBRT集成：左侧拟合训练集的树数量不足，而右侧拟合训练集的树数量过多从而导致过拟合。

![fig14_GBRT集成](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig14_GBRT%E9%9B%86%E6%88%90.jpg)

要找到树的最佳数量，可以使用提前停止法（参见第4章）。简单的实现方法就是使用`staged_predict()`方法：它在训练的每个阶段（一棵树时，两棵树时，等等）都对集成的预测返回一个迭代器。以下代码训练了一个拥有120棵树的GBRT集成，然后测量每个训练阶段的验证误差，从而找到树的最优数量，最后使用最优树数重新训练了一个GBRT集成：

```python
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
    gbrt.fit(X_train, y_train)

    errors = [mean_squared_error(y_val, y_pred)
            for y_pred in gbrt.staged_predict(X_val)]
    bst_n_estimators = np.argmin(errors) + 1

    gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
    gbrt_best.fit(X_train, y_train)
```

验证误差显示在图15的左侧，最佳模型的预测显示在右侧。

![fig15_通过提前停止法调整树的数量](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig15_%E9%80%9A%E8%BF%87%E6%8F%90%E5%89%8D%E5%81%9C%E6%AD%A2%E6%B3%95%E8%B0%83%E6%95%B4%E6%A0%91%E7%9A%84%E6%95%B0%E9%87%8F.jpg)

实际上，要实现提前停止法，不一定需要先训练大量的树，然后再回头找最优的数字，还可以提前停止训练。设置`warm_start=True`，当`fit()`方法被调用时，Scikit-Learn会保留现有的树，从而允许增量训练。以下代码会在验证误差连续5次迭代未改善时，直接停止训练：

```python
    gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)

    min_val_error = float("inf")
    error_going_up = 0
    for n_estimators in range(1, 120):
        gbrt.n_estimators = n_estimators
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_val)
        val_error = mean_squared_error(y_val, y_pred)
        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up == 5:
                break  # early stopping
```

`GradientBoostingRegressor`类还可以支持超参数`subsample`，指定用于训练每棵树的实例的比例。例如，如果`subsample=0.25`，则每棵树用25%的随机选择的实例进行训练。现在你可以猜到，这也是用更高的偏差换取了更低的方差，同时在相当大的程度上加速了训练过程。这种技术被称为**随机梯度提升**。

值得注意的是，流行的Python库XGBoost（该库代表Extreme Gradient Boosting）中提供了梯度提升的优化实现，该软件包最初是由Tianqi Chen作为分布式（深度）机器学习社区（DMLC）的一部分开发的，其开发目标是极快、可扩展和可移植。实际上，XGBoost通常是ML竞赛中获胜的重要组成部分。XGBoost的API与Scikit-Learn的非常相似。

## 7.6 堆叠法

本章我们要讨论的最后一个集成方法叫作**堆叠法（stacking）**，又称**层叠泛化法**。它基于一个简单的想法：与其使用一些简单的函数（比如硬投票）来聚合集成中所有预测器的预测，我们为什么不训练一个模型来执行这个聚合呢？图16显示了在新实例上执行回归任务的这样一个集成。底部的三个预测器分别预测了不同的值（3.1、2.7和2.9），然后最终的预测器（称为混合器或元学习器）将这些预测作为输入，进行最终预测（3.0）。

![fig16_通过混合预测器聚合预测](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig16_%E9%80%9A%E8%BF%87%E6%B7%B7%E5%90%88%E9%A2%84%E6%B5%8B%E5%99%A8%E8%81%9A%E5%90%88%E9%A2%84%E6%B5%8B.jpg)

训练混合器的常用方法是使用**留存集**。我们看看它是如何工作的。首先，将训练集分为两个子集，第一个子集用来训练第一层的预测器（见图17）。

![fig17_训练第一层](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig17_%E8%AE%AD%E7%BB%83%E7%AC%AC%E4%B8%80%E5%B1%82.jpg)

然后，用第一层的预测器在第二个（留存）子集上进行预测（见图18）。因为预测器在训练时从未见过这些实例，所以可以确保预测是“干净的”。那么现在对于留存集中的每个实例都有了三个预测值。我们可以使用这些预测值作为输入特征，创建一个新的训练集（新的训练集有三个维度），并保留目标值。在这个新的训练集上训练混合器，让它学习根据第一层的预测来预测目标值。

![fig18_训练混合器](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig18_%E8%AE%AD%E7%BB%83%E6%B7%B7%E5%90%88%E5%99%A8.jpg)

事实上，通过这种方法可以训练多种不同的混合器（例如，一个使用线性回归，另一个使用随机森林回归，等等）。于是我们可以得到一个混合器层。诀窍在于将训练集分为三个子集：**第一个用来训练第一层，第二个用来创造训练第二层的新训练集（使用第一层的预测），而第三个用来创造训练第三层的新训练集（使用第二层的预测）**。一旦训练完成，我们可以按照顺序遍历每层来对新实例进行预测，如图19所示。

![fig19_一个多层堆叠集成的预测](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter7/figures/fig19_%E4%B8%80%E4%B8%AA%E5%A4%9A%E5%B1%82%E5%A0%86%E5%8F%A0%E9%9B%86%E6%88%90%E7%9A%84%E9%A2%84%E6%B5%8B.jpg)

不幸的是，Scikit-Learn不直接支持堆叠，但是推出自己的实现并不太难（参见接下来的练习题）。或者，你也可以使用开源的实现方案，例如DESlib。