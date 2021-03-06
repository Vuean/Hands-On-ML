# 第十章 Keras人工神经网络简介

人工神经网络是深度学习的核心。它们用途广泛、功能强大且可扩展，使其**非常适合处理大型和高度复杂的机器学习任务**，例如对数十亿张图像进行分类（例如Google Images），为语音识别服务（例如Apple的Siri）提供支持，每天向成千上万的用户推荐（例如YouTube）观看的最佳视频，或学习在围棋游戏（DeepMind的AlphaGo）中击败世界冠军。

本章的第一部分介绍了人工神经网络，首先是对第一个ANN架构的快速浏览，然后是今天广泛使用的多层感知机（MLP）（其他架构将在第11章中进行探讨）。在第二部分中，我们将研究如何使用流行的Keras API实现神经网络。这是设计精巧、简单易用的用于构建、训练、评估和运行神经网络的API。但是，不要被它的简单性所迷惑：它的表现力和灵活性足以让你构建各种各样的神经网络架构。实际上，对于大多数示例而言，这可能就足够了。如果你需要额外的灵活性，可以随时使用其较低级的API编写自定义的Keras组件，这将在第12章中讨论。

## 10.1 从生物神经元到人工神经元

McCulloch和Pitts在其具有里程碑意义的论文“*A Logical Calculus of Ideas Immanent in Nervous Activity*”中，提出了一种简化的计算模型，该模型计算了生物神经元如何在动物大脑中协同工作，利用命题逻辑进行复杂的计算。这是第一个人工神经网络架构。

人们对人工神经网络重新充满兴趣将对我们的生活产生更深远的影响：

- 现在有大量数据可用于训练神经网络，并且在非常大和复杂的问题上，人工神经网络通常优于其他机器学习技术。

- 自20世纪90年代以来，计算能力的飞速增长使得现在有可能在合理的时间内训练大型神经网络。

- 训练算法已得到改进。

- 在实践中，人工神经网络的一些理论局限性被证明是良性的。例如，许多人认为ANN训练算法注定要失败，因为它们可能会陷入局部最优解，但事实证明，这在实践中相当罕见。

- 人工神经网络似乎已经进入了资金和发展的良性循环。

### 10.1.1 生物神经元

生物神经元产生短的电脉冲称为动作电位（AP，或只是信号），它们沿着轴突传播，使突触释放称为神经递质的化学信号。当神经元在几毫秒内接收到足够数量的这些神经递质时，它会激发自己的电脉冲。

![fig01_生物神经元](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig01_%E7%94%9F%E7%89%A9%E7%A5%9E%E7%BB%8F%E5%85%83.jpg)

因此，单个生物神经元的行为似乎很简单，但是它们组成了数十亿个庞大的网络，每个神经元都与数千个其他神经元相连。高度复杂的计算可以通过相当简单的神经元网络来执行，就像复杂的蚁丘可以通过简单蚂蚁的共同努力而出现一样。生物神经网络（BNN）的架构仍是活跃的研究主题，但大脑的某些部分已被绘制成图，似乎神经元通常组织成连续的层，尤其是在大脑皮层中（大脑的外层），如图2所示。

![fig02_生物神经网络](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig02_%E7%94%9F%E7%89%A9%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.jpg)

### 10.1.2 神经元的逻辑计算

McCulloch和Pitts提出了一个非常简单的生物神经元模型，该模型后来被称为**神经元(artificial neuron)**。它具有*一个或多个二进制（开/关）输入和一个二进制输出*。当超过一定数量的输入处于激活状态时，人工神经元将激活其输出。他们的论文表明即使使用这样的简化模型，也可以构建一个人工神经元网络来计算所需的任何逻辑命题。为了了解这种网络的工作原
理，让我们构建一些执行各种逻辑计算的ANN（见图3），假设神经元至少两个输入处于激活状态时，神经元就会被激活。

![fig03_ANN执行简单的逻辑运算](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig03_ANN%E6%89%A7%E8%A1%8C%E7%AE%80%E5%8D%95%E7%9A%84%E9%80%BB%E8%BE%91%E8%BF%90%E7%AE%97.jpg)

网络的作用如下所述：

- 左边的第一个网络是恒等函数：如果神经元A被激活，那么神经元C也被激活（因为它从神经元A接收到两个输入信号）；如果神经元A关闭，那么神经元C也关闭。

- 第二个网络执行逻辑AND：仅当神经元A和B都被激活（单个输入信号不足以激活神经元C）时，神经元C才被激活。

- 第三个网络执行逻辑OR：如果神经元A或神经元B被激活（或两者都激活），则神经元C被激活。

- 最后，如果我们假设输入连接可以抑制神经元的活动（生物神经元就是这种情况），则第四个网络计算出一个稍微复杂的逻辑命题：只有在神经元A处于活动状态和神经元B关闭时，神经元C才被激活。如果神经元A一直处于活动状态，那么你会得到逻辑NOT：神经元B关闭时神经元C处于活动状态，反之亦然。

### 10.1.3 感知器

**感知器(Perceptron)**是最简单的ANN架构之一，由Frank Rosenblatt于1957年发明。它基于稍微不同的人工神经元（见图4），称为**阈值逻辑单元（TLU）**，有时也称为**线性阈值单元（LTU）**。输入和输出是数字（而不是二进制开/关值），并且每个输入连接都与权重相关联。TLU计算其输入的加权和（ $z=w_1x_1+w_2x_2+...+w_nx_n=x^Tw$），然后将**阶跃函数**应用于该和并输出结果：$h_w(x)=step(z)$，其中 $z=x^Tw$。

![fig04_阈值逻辑单元](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig04_%E9%98%88%E5%80%BC%E9%80%BB%E8%BE%91%E5%8D%95%E5%85%83.jpg)

感知器中最常用的阶跃函数是**Heaviside阶跃函数**（见公式10-1）。有时使用符号函数代替。

![fig05_公式10-1_Heaviside阶跃函数](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig05_%E5%85%AC%E5%BC%8F10-1_Heaviside%E9%98%B6%E8%B7%83%E5%87%BD%E6%95%B0.jpg)

单个TLU可用于简单的线性二进制分类。它计算输入的线性组合，如果结果超过阈值，则输出正类；否则，它将输出负类（就像逻辑回归或线性SVM分类器一样）。例如，你可以使用单个TLU根据花瓣的长度和宽度对鸢尾花进行分类（就像我们在前面的章节中所做的那样，还添加了额外的偏置特征 $x_0=1$）。在这种情况下，训练TLU意味着找到$w_0$、$w_1$和$w_2$的正确值（稍后将讨论训练算法）。

**感知器仅由单层TLU组成**，每个TLU连接到所有的输入。当一层中的所有神经元都连接到上一层中的每个神经元（即其输入神经元）时，该层称为**全连接层或密集层**。感知器的输入被送到称为**输入神经元**的特殊直通神经元：它们输出被送入的任何输入。**所有输入神经元形成输入层**。此外，通常会添加一个额外的偏置特征（$x_0=1$）：通常使用一种称为**偏置神经元**的特殊类型的神经元来表示该特征，该神经元始终输出1。具有两个输入和三个输出的感知器如图5所示。**该感知器可以将实例同时分为三个不同的二进制类，这使其成为多输出分类器**。

![fig06_感知器架构](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig06_%E6%84%9F%E7%9F%A5%E5%99%A8%E6%9E%B6%E6%9E%84.jpg)

借助线性代数的魔力，公式10-2使得可以同时为多个实例高效地计算出一层人工神经元的输出。

![fig07_计算全连接层的输出](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig07_%E8%AE%A1%E7%AE%97%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82%E7%9A%84%E8%BE%93%E5%87%BA.jpg)

其中，

- X代表输入特征的矩阵。每个实例一行，每个特征一列。

- 权重矩阵W包含除偏置神经元外的所有连接权重。在该层中，每个输入神经元一行，每个人工神经元一列。

- 偏置向量b包含偏置神经元和人工神经元之间的所有连接权重。每个人工神经元有一个偏置项。

- 函数φ称为**激活函数**：当人工神经元是TLU时，它是阶跃函数（但我们在后面会讨论其他激活函数）。

那么，感知器如何训练？Rosenblatt提出的感知器训练算法在很大程度上受Hebb规则启发。Donald Hebb在其1949年的The Organization of Behavior（Wiley）中提出，当一个生物神经元经常触发另一个神经元时，这两个神经元之间的联系就会增强。后来，Siegrid Löwel用有名的措辞概括了Hebb的思想，即“触发的细胞，连接在一起”。也就是说，**两个神经元同时触发时，它们之间的连接权重会增加**。该规则后来被称为Hebb规则（或Hebb学习）。使用此规则的变体训练感知器，该变体考虑了网络进行预测时所犯的错误。感知器学习规则加强了有助于减少错误的连接。更具体地说，感知器一次被送入一个训练实例，并且针对每个实例进行预测。对于产生错误预测的每个输出神经元，它会增强来自输入的连接权重，这些权重将有助于正确的预测。该规则如公式3所示。

![fig08_公式10-3_感知器学习规则](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig08_%E5%85%AC%E5%BC%8F10-3_%E6%84%9F%E7%9F%A5%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%A7%84%E5%88%99.jpg)

其中：

- $w_{i,j}$是第i个输入神经元和第j个输出神经元之间的连接权重；

- $x_i$是当前训练实例的第i个输入值；

- $\haty_j$是当前训练实例的第j个输出神经元的输出。

- $y_j$是当前训练实例的第j个输出神经元的目标输出。

- $η$是学习率。

每个**输出神经元的决策边界都是线性的**，因此感知器无法学习复杂的模式（就像逻辑回归分类器一样）。但是，如果训练实例是线性可分的，osenblatt证明了该算法将收敛到一个解。这被称为**感知器收敛定理**。

Scikit-Learn提供了一个Perceptron类，该类实现了单个TLU网络。它可以像你期望的那样使用，例如，在鸢尾植物数据集上：

```python
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.linear_model import Perceptron

    iris = load_iris()
    X = iris.data[:, (2, 3)]  # petal length, petal width
    y = (iris.target == 0).astype(np.int)

    per_clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    per_clf.fit(X, y)

    y_pred = per_clf.predict([[2, 0.5]])
```
你可能已经注意到，感知器学习算法非常类似于随机梯度下降。实际上，Scikit-Learn的Perceptron类等效于使用具有以下超参数的 `SGDClassifier: loss="perceptron"`，`learning_rate="constant"`，`eta0=1`（学习率）和`penalty=None`（无正则化））。请注意，与逻辑回归分类器相反，感知器不输出分类概率；相反，它们基于硬阈值进行预测。这是逻辑回归胜过感知器的原因。

Marvin Minsky和Seymour Papert在1969年的专著Perceptron中，特别指出了感知器的一些严重缺陷，即它们无法解决一些琐碎的问题（例如，异或（XOR）分类问题，参见图9的左侧）。任何其他线性分类模型（例如逻辑回归分类器）都是如此，但是研究人员对感知器的期望更高，有些人感到失望，他们完全放弃了神经网络，转而支持更高层次的问题，例如逻辑、问题求解和搜索。

![fig09_XOR分类问题和解决该问题的MLP](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig09_XOR%E5%88%86%E7%B1%BB%E9%97%AE%E9%A2%98%E5%92%8C%E8%A7%A3%E5%86%B3%E8%AF%A5%E9%97%AE%E9%A2%98%E7%9A%84MLP.jpg)

事实证明，可以通过堆叠多个感知器来消除感知器的某些局限性。所得的ANN称为**多层感知器（MLP）**。MLP可以解决XOR问题，你可以通过计算图9右侧所示的MLP的输出来验证：输入(0,0)或(1,1)，网络输出0，输入(0,1)或(1,0)则输出1。所有连接的权重等于1，但显示权重的四个连接除外。尝试验证该网络确实解决了XOR问题！

### 10.1.4 多层感知器和反向传播

MLP由一层（直通）输入层、一层或多层TLU（称为隐藏层）和一个TLU的最后一层（称为输出层）组成（见图10）。靠近输入层的层通常称为**较低层**，靠近输出层的层通常称为**较高层**。除输出层外的每一层都包含一个偏置神经元，并完全连接到下一层。

信号仅沿一个方向（从输入到输出）流动，因此该架构是**前馈神经网络（FNN）**的示例。

![fig10_多层感知器架构](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig10_%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E5%99%A8%E6%9E%B6%E6%9E%84.jpg)

当一个ANN包含一个深层的隐藏层时，它称为**深度神经网络（DNN）**。深度学习领域研究DNN，更广泛地讲包含深度计算堆栈的模型。即便如此，只要涉及神经网络（甚至是浅层的神经网络），许多人就会谈论深度学习。

1986年，David Rumelhart、Geoffrey Hinton和Ronald Williams发表的开创性论文介绍了**反向传播训练算法**，该算法至今仍在使用。简而言之，它是使用有效的技术自动计算梯度下降（在第4章中介绍）：在仅两次通过网络的过程中（一次前向，一次反向），反向传播算法能够针对每个模型参数计算网络误差的梯度。换句话说，它可以找出应如何调整每个连接权重和每个偏置项以减少误差。一旦获得了这些梯度，它便会执行常规的梯度下降步骤，然后重复整个过程，直到网络收敛到解。

自动计算梯度称为自动微分或者autodiff。有各种autodiff技术，各有优缺点。反向传播使用的一种称为反向模式autodiff。它快速而精确，并且非常适用于微分函数具有多个变量（例如，连接权重）和少量输出（例如，一个损失）的情况。

让我们更详细地介绍一下该算法：

- 它一次处理一个小批量（例如，每次包含32个实例），并且多次遍历整个训练集。每次遍历都称为一个**轮次**。

- 每个小批量都传递到网络的输入层，然后将其送到第一个隐藏层。然后该算法将计算该层中所有神经元的输出（对于小批量中的每个实例）。结果传递到下一层，计算其输出并传递到下一层，以此类推，直到获得最后一层（即输出层）的输出。这就是**前向通路**：就像进行预测一样，只是保留了所有中间结果，因为反向遍历需要它们。

- 接下来，该算法测量网络的输出误差（该算法使用一种损失函数，该函数将网络的期望输出与实际输出进行比较，并返回一些误差测量值）。

- 然后，它计算每个输出连接对错误的贡献程度。通过应用链式法则（可能是微积分中最基本的规则）来进行分析，从而使此步骤变得快速而精确。

- 然后，算法再次使用链式法则来测量这些错误贡献中有多少是来自下面层中每个连接的错误贡献，算法一直进行，到达输入层为止。如前所述，这种反向传递通过在网络中向后传播误差梯度，从而有效地测量了网络中所有连接权重上的误差梯度（因此称为算法）。

- 最终，该算法执行梯度下降步骤，使用刚刚计算出的误差梯度来调整网络中的所有连接权重。

该算法非常重要，值得再次总结：对于每个训练实例，反向传播算法首先进行预测（正向传递）并测量误差，然后反向经过每个层以测量来自每个连接的误差贡献（反向传递），最后调整连接权重以减少错误（梯度下降步骤）。

随机初始化所有隐藏层的连接权重很重要，否则训练将失败。例如，如果将所有权重和偏置初始化为零，则给定层中的所有神经元将完全相同，从而反向传播将以完全相同的方式影响它们，因此它们将保持相同。换句话说，尽管每层有数百个神经元，但是模型会像每层只有一个神经元一样工作：不会太聪明。相反，如果随机初始化权重，则会破坏对称性，并允许反向传播来训练各种各样的神经元。

为了使该算法正常工作，作者对MLP的架构进行了重要更改，将阶跃函数替换为逻辑（s型）函数：σ(z)=1/(1+exp(-z))。这一点很重要，因为阶跃函数仅包含平坦段，所以没有梯度可使用（梯度下降不能在平面上移动），而逻辑函数在各处均具有定义明确的非零导数，从而使梯度下降在每一步都可以有所进展。实际上，反向传播算法可以与许多其他激活函数（不仅是逻辑函数）一起很好地工作。这是另外两个受欢迎的选择：

双曲正切函数：tanh(z)=2σ(2z)-1

与逻辑函数一样，该激活函数为S形、连续且可微，但其输出值范围为-1到1（而不是逻辑函数的从0到1）。在训练开始时，该范围倾向于使每一层的输出或多或少地以0为中心，这通常有助于加快收敛速度。

线性整流单位函数：ReLU(z)=max(0，z)

ReLU函数是连续的，但不幸的是，在z=0时，该函数不可微分（斜率会突然变化，这可能使梯度下降反弹），如果z<0则其导数为0。但是，实际上它运行良好并且具有计算快速的优点，因此它已成为默认值。最重要的是，它没有最大输出值这一事实有助于减少梯度下降期间的某些问题（我们将在第11章中对此进行讨论）。

这些流行的激活函数及其派生函数如图11所示。为什么我们首先需要激活函数？如果连接多个线性变换，那么得到的只是一个线性变换。例如，如果f(x)=2x+3且g(x)=5x-1，则连接这两个线性函数可以得到另一个线性函数：f(g(x))=2(5x-1）+3=10x+1。因此，如果层之间没有非线性，那么即使是很深的层堆叠也等同于单个层，这样你无法解决非常复杂的问题。相反，具有非线性激活函数的足够大的DNN理论上可以近似任何连续函数。

![fig11_激活函数及其派生](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig11_%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%8F%8A%E5%85%B6%E6%B4%BE%E7%94%9F.jpg)

好！你知道神经网络来自何处、其结构是什么以及如何计算其输出。你还了解了反向传播算法。但是，你到底可以使用它们做什么呢？

### 10.1.5 回归MLP

首先，MLP可用于回归任务。如果要预测单个值（房屋的价格，给定其许多特征），则只需要单个输出神经元：其输出就是预测值。对于多元回归（即一次预测多个值），每个输出维度需要一个输出神经元。例如，要在图像中定位物体的中心，你需要预测2D坐标，因此需要两个输出神经元。如果你还想在物体周围放置边框，则还需要两个数字：

物体的宽度和高度。因此，你得到了四个输出神经元。

通常，在构建用于回归的MLP时，你不想对输出神经元使用任何激活函数，因此它们可以输出任何范围的值。如果要保证输出始终为正，则可以在输出层中使用ReLU激活函数。另外，你可以使用`softplus`激活函数，它是ReLU的平滑变体：softplus(z)=log(1+exp(z))。当z为负时，它接近于0，而当z为正时，它接近于z。最后，如果要保证预测值落在给定的值范围内，则可以使用逻辑函数或双曲正切，然后将标签缩放到适当的范围：逻辑函数的范围为0到1，双曲正切为-1到1。

训练期间要使用的损失函数通常是均方误差，但是如果训练集中有很多离群值，则你可能更愿意使用平均绝对误差。或者，你可以使用Huber损失，这是两者的组合。

当误差小于阈值δ（通常为1）时，Huber损失为二次方，而当误差大于δ时，Huber损失为线性。线性部分使它对离群值的敏感性低于均方误差，而二次方部分使它比平均绝对误差更收敛并且更精确。

表10-1总结了回归MLP的典型架构。

|超参数|典型值|
|:---:|:---:|
|输入神经元数量|每个输入特征一个|
|隐藏层数量|取决于问题，但通常为1到5|
|每个隐藏层的神经元数量|取决于问题，但通常为10到100|
|输出神经元数量|每个预测维度输出1个神经元|
|隐藏的激活|ReLU（或SELU，见第11章）|
|输出激活|无，或ReLU/softplus（如果为正输出）或逻辑/tanh（如果为有界输出）|
|损失函数|MSE或MAE/Huber（如果存在离群值）|

### 10.1.6 分类MLP

MLP也可以用于分类任务。对于二进制分类问题，你只需要使用逻辑激活函数的单个输出神经元：输出将是0到1之间的数字，你可以将其解释为正类的估计概率。负类别的估计概率等于一减去该数字。

MLP还可以轻松处理多标签二进制分类任务（第3章）。例如，你可能有一个电子邮件分类系统，该系统可以预测每个收到的电子邮件是正常邮件还是垃圾邮件，并同时预测它是紧急电子邮件还是非紧急电子邮件。在这种情况下，你需要两个输出神经元，两个都使用逻辑激活函数：第一个输出电子邮件为垃圾邮件的可能性，第二个输出紧急邮件的可能性。更一般地，你为每个正类别用一个输出神经元。请注意，输出概率不一定要加起来为1。这可以使模型输出任何组合的标签：你可以包含非紧急正常邮件、紧急正常邮件、非紧急垃圾邮件，甚至可能是紧急垃圾邮件（尽管可能是一个错误）。

如果每个实例只能属于三个或更多可能的类中的一个类（例如，用于数字图像分类的类0到9），则每个类需要一个输出神经元，并且应该使用softmax激活函数整个输出层（见图12）。softmax函数（在第4章中介绍）将确保所有估计的概率在0到1之间，并且它们加起来等于1（如果类是互斥的，则是必需的）。这称为多类分类。

![图fig12_用于分类的现代MLP](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig12_%E7%94%A8%E4%BA%8E%E5%88%86%E7%B1%BB%E7%9A%84%E7%8E%B0%E4%BB%A3MLP.jpg)

关于损失函数，由于我们正在预测概率分布，因此交叉熵损失（也称为对数损失，见第4章）通常是一个不错的选择。表10-2总结了分类MLP的典型架构：

表10-2：典型的MLP架构

|超参数|二进制分类|多标签二进制分类|多类分类|
|:---:|:---:|:---:|:---:|
|输入层和隐藏层|与回归相同|与回归相同|与回归相同|
|输出神经元数量|1|每个标签1|每个类1|
|输出层激活|逻辑|逻辑|softmax|
|损失函数|交叉熵|交叉熵|交叉熵|

## 10.2 使用Keras实现MLP

Keras是高级深度学习API，可让你轻松构建、训练、评估和执行各种神经网络。其文档（或规范）可从https://keras.io/获得。这个参考实现也称为Keras，由Fransois Chollet开发，是一个研究项目的一部分，并于2015年3月作为开源项目发布。由于其易用性、灵活性和精巧设计，它迅速流行。为了执行神经网络所需的繁重计算，此参考实现依赖于计算后端。目前你可以从三种流行的开源深度学习库中进行选择：TensorFlow、微软的Cognitive Toolkit（CNTK）和Theano。因此，为避免混淆，我们将此参考实现称为多后端Keras。

自2016年底以来发布了其他的实现。现在，你可以在Apache MXNet、苹果的Core ML、avaScript或TypeScript（可以在网络浏览器中运行Keras代码）和PlaidML（可以在各种GPU设备上运行，而不仅仅是Nvidia）上运行Keras。而且，TensorFlow本身现在与自己的Keras实现程序tf.keras捆绑在一起。它仅支持TensorFlow作为后端，但具有提供一些有用的额外功能的优势（见图13）：例如，它支持TensorFlow的数据API，可轻松高效地加载和预处理数据。因此，我们将在本书中使用tf.keras。但是，在本章中，我们将不会使用任何特定于TensorFlow的功能，因此该代码也应该可以在其他Keras实现上很好地运行（至少在Python中），并且只需进行少量修改即可，例如更改导入。

![fig13_Keras API的两种实现](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig13_Keras%20API%E7%9A%84%E4%B8%A4%E7%A7%8D%E5%AE%9E%E7%8E%B0.jpg)

在Keras和TensorFlow之后，最受欢迎的深度学习库是Facebook的PyTorch库。好消息是它的API与Keras的API十分相似（部分原因是这两个API均受Scikit-Learn和Chainer的启发），因此一旦你了解Keras，便可以轻松切换到PyTorch（如果你想的话）。PyTorch的受欢迎程度在2018年呈指数增长，这主要归功于它的简单性和出色的文档，而这并不是TensorFlow 1.x的主要优势。但是TensorFlow 2可以说与PyTorch一样简单，因为它采用Keras作为其官方高级API，并且其开发人员简化和清理了其余的API。该文档也已被完全重新组织，现在更容易找到所需的内容。同样，PyTorch1.0的主要缺点（例如，有限的可移植性和无计算图分析）已得到解决。健康的竞争对所有人都有利。

好了，该写代码了！由于tf.keras与TensorFlow捆绑在一起，让我们从安装TensorFlow开始。

### 10.2.1 安装Tensorflow2

### 10.2.2 使用顺序API构建图像分类器

首先，我们需要加载数据集。在本章中，我们将介绍Fashion MNIST，它是MNIST的直接替代品（在第3章中介绍）。它具有与MNIST完全相同的格式（70 000张灰度图像，每幅28×28像素，有10个类），但是这些图像代表的是时尚物品，而不是手写数字，因此每个类更有多样性，问题比MNIST更具挑战性。例如，简单的线性模型在MNIST上达到约92％的准确率，但在Fashion MNIST上仅达到约83％的准确率。

使用Keras加载数据集

Keras提供了一些实用程序来获取和加载常见数据集，包括MNIST、Fashion MNIST和我们在第2章中使用的加州房屋数据集。让我们加载Fashion MNIST：

```python
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
```

当使用Keras而不是Scikit-Learn来加载MNIST或Fashion MNIST时，一个重要的区别是每个图像都表示为28×28阵列，而不是尺寸为784的一维阵列。此外，像素强度表示为整数（从0到255）而不是浮点数（从0.0到255.0）。让我们看一下训练集的形状和数据类型：

```python
    X_train_full.shape
    >>> (60000, 28, 28)
    X_train_full.dtype
    >>> dtype('uint8')
```

请注意，数据集已经分为训练集和测试集，但是没有验证集，因此我们现在创建一个。另外，由于我们要使用梯度下降训练神经网络，因此必须比例缩放输入特征。为了简单起见，我们将像素强度除以255.0（将它们转换为浮点数），将像素强度降低到0～1范围内：

```python
    X_valid, X_train = X_train_full[:5000]/255., X_train_full[5000:]/255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.
```

对于MNIST，当标签等于5时，说明图像代表手写数字5。但是，对于Fashion MNIST，我们需要一个类名列表来知道我们要处理的内容：

```python
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

例如，训练集中的第一幅图像代表一件外套：

```python
    class_names[y_train[0]]
    >>> 'Coat'
```

图14显示了来自Fashion MNIST数据集的一些示例。

![fig14_Fashion MNIST的样本](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig14_Fashion%20MNIST%E7%9A%84%E6%A0%B7%E6%9C%AC.jpg)

#### 使用顺序API创建模型

现在让我们建立神经网络！这是具有两个隐藏层的分类MLP：

```python
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
```

让我们逐行浏览以下代码：

- 第一行创建一个Sequential模型。这是用于神经网络的最简单的Keras模型，它仅由顺序连接的单层堆栈组成。这称为顺序API。

- 接下来，我们构建第一层并将其添加到模型中。它是**Flatten层**，其作用是将每个输入图像转换为一维度组：如果接收到输入数据X，则计算X.reshape(-1,1)。该层没有任何参数。它只是在那里做一些简单的预处理。由于它是模型的第一层，因此应指定`input_shape`，其中不包括批处理大小，而仅包括实例的形状。或者，你可以添加`keras.layers.InputLayer`作为第一层，设置`input_shape=[28,28]`。

- 接下来，我们添加具有300个神经元的Dense隐藏层。它使用ReLU激活函数。每个Dense层管理自己的权重矩阵，其中包含神经元及其输入之间的所有连接权重。它还管理偏置项的一个向量（每个神经元一个）。当它接收到一些输入数据时，它计算公式10-2。

- 然后，我们添加第二个有100个神经元的Dense隐藏层，还是使用ReLU激活函数。

- 最后，我们添加一个包含10个神经元的Dense输出层（每个类一个），使用softmax激活函数（因为这些类是排他的）。

指定`activation="relu"`等效于指定`activation=keras.activations.relu`。keras.activations软件包中提供了其他激活函数，我们将在本书中使用其中的许多函数。有关完整列表，请参见https://keras.io/activations/。

可以不用像我们刚才那样逐层添加层，而可以在创建顺序模型时传递一个层列表：

```python
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
```

模型的`summary()`方法显示模型的所有层，包括每个层的名称（除非在创建层时进行设置，否则会自动生成），其输出形状（None表示批处理大小任意），以及它的参数数量。总结以参数总数结尾，包括可训练参数和不可训练的参数。在这里，我们只有可训练的参数（我们将在第11章中看到不可训练参数的示例）:

```python
    model.summary()
    >>> Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 300)               235500    
_________________________________________________________________
dense_1 (Dense)              (None, 100)               30100     
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1010      
=================================================================
Total params: 266,610
Trainable params: 266,610
Non-trainable params: 0
_________________________________________________________________
```

请注意，密集层通常具有很多参数。例如，第一个隐藏层的连接权重为784×300，外加300个偏置项，总共有235500个参数！这为模型提供了足够的灵活性来拟合训练数据，但这也意味着模型存在过拟合的风险，尤其是在你没有大量训练数据的情况下。我们稍后会再谈。

你可以轻松获取模型的层列表，按其索引获取层，也可以按名称获取：

```python
    model.layers
    >>> 
    [<tensorflow.python.keras.layers.core.Flatten at 0x1ad9f5270a0>,
    <tensorflow.python.keras.layers.core.Dense at 0x1ad9f5274f0>,
    <tensorflow.python.keras.layers.core.Dense at 0x1ad9f527be0>,
    <tensorflow.python.keras.layers.core.Dense at 0x1ad9f527f70>]

    hidden1 = model.layers[1]
    hidden1.name
    >>> 'dense'

    model.get_layer(hidden1.name) is hidden1
    >>> True
```

可以使用`get_weights()`和`set_weights()`方法访问层的所有参数。对于密集层，这包括连接权重和偏置项：

```python
    weights, biases = hidden1.get_weights()
    weights
    >>> 
    array([[ 0.02448617, -0.00877795, -0.02189048, ..., -0.02766046,
         0.03859074, -0.06889391],
       [ 0.00476504, -0.03105379, -0.0586676 , ...,  0.00602964,
        -0.02763776, -0.04165364],
       [-0.06189284, -0.06901957,  0.07102345, ..., -0.04238207,
         0.07121518, -0.07331658],
       ...,
       [-0.03048757,  0.02155137, -0.05400612, ..., -0.00113463,
         0.00228987,  0.05581069],
       [ 0.07061854, -0.06960931,  0.07038955, ..., -0.00384101,
         0.00034875,  0.02878492],
       [-0.06022581,  0.01577859, -0.02585464, ..., -0.00527829,
         0.00272203, -0.06793761]], dtype=float32)

    weights.shape
    >>> (784, 300)

    biases
    >>> 
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    ...], dtype=float32)

    biases.shape
    >>> (300,)
```

注意，密集层随机初始化了连接权重（这是打破对称性所必需的，正如我们前面所讨论的），并且偏置被初始化为零，这是可以的。如果要使用其他初始化方法，则可以在创建层时设置`kernel_initializer`（内核是连接权重矩阵的另一个名称）或`bias_initializer`。我们将在第11章中进一步讨论初始化，但是如果需要完整的列表，请参见https://keras.io/initializers/。

权重矩阵的形状取决于输入的个数。这就是在`Sequential`模型中创建第一层时建议指定`input_shape`的原因。但是，如果你不指定输入形状，那也是可以的：Keras会等到知道输入形状后才真正构建模型。当你向其提供实际数据时（例如，在训练期间），或者在调用其`build()`方法时，就会发生这种情况。在真正构建模型之前，所有层都没有权重，而且你也无法执行某些操作（例如打印模型总结或保存模型）。因此，如果在创建模型时知道输入形状，则最好指定它。

#### 编译模型

创建模型后，你必须调用`compile()`方法来指定损失函数和要使用的优化器。也可以选择指定在训练和评估期间要计算的其他指标：

```python
    model.compile(loss="sparse_categorical_crossentropy",
        optimizer="sgd", metrics=["accuracy"])
```

使用`loss="sparse_categorical_crossentropy"`等同于使用`loss=keras.losses.sparse_categorical_crossentropy`。同样，指定`optimizer="sgd"`等同于指定`optimizer=keras.optimizers.SGD()`，而`metrics=["acc uracy"]`等同于`metrics=[keras.metrics.sparse_categori cal_accuracy]`（使用此损失时）。在本书中，我们将使用许多其他的损失、优化器和指标。有关完整列表，请参见https://keras.io/losses、https://keras.io/optimizers和https://keras.io/metrics。

此代码需要一些解释。首先，我们使用"sparse_categorical_crossentropy"损失，因为我们具有稀疏标签（即对于每个实例，只有一个目标类索引，在这种情况下为0到9），并且这些类是互斥的。相反，如果每个实例的每个类都有一个目标概率（例如独热向量，[0.，0.，0.，1.，0.，0.，0.，0.，0.，0]代表类3），则我们需要使
用"categorical_crossentropy"损失。如果我们正在执行二进制分类（带有一个或多个二进制标签），则在输出层中使用"sigmoid"（即逻辑）激活函数，而不是"softmax"激活函数，并且使用"binary_crossentropy"损失。

如果要将稀疏标签（即类索引）转换为独热向量标签，使用`keras.utils.to_categorical()`函数。反之则使用`np.argmax()`函数和`axis=1`。

关于优化器，"sgd"表示我们使用简单的随机梯度下降来训练模型。换句话说，Keras将执行先前所述的反向传播算法（即反向模式自动微分加梯度下降）。我们将在第11章中讨论更有效的优化器（它们改进梯度下降部分，而不是自动微分）。

使用SGD优化器时，调整学习率很重要。因此通常需要使用`optimizer=keras.optimizers.SGD(lr=???)`来设置学习率，而不是使用`optimizer="sgd"`（默
认值为lr=0.01）来设置学习率。

最后，由于这是一个分类器，因此在训练和评估过程中测量其"accuracy"很有用。

#### 训练和评估模型

现在该模型已准备好进行训练。为此我们只需要调用其`fit()`方法即可：

```python
    history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
```

我们将输入特征（X_train）和目标类（y_train）以及要训练的轮次数传递给它（否则它将默认为1，这绝对不足以收敛为一个好的模型）。我们还传递了一个验证集（这是可选的）。Keras将在每个轮次结束时测量此集合上的损失和其他指标，这对于查看模型的实际效果非常有用。如果训练集的性能好于验证集，则你的模型可能过拟合训练集（或者存在错误，例如训练集和验证集之间的数据不匹配）。

就是这样！训练了神经网络注。在训练期间的每个轮次，Keras会显示到目前为止已处理的实例数（以及进度条）、每个样本的平均训练时间、损失和精度（或你要求的任何其他额外指标）（针对训练集和验证集）。你可以看到训练损失减少了，这是一个好兆头，经过30个轮次后，验证准确率达到了89.26％。这与训练精确率相差不大，因此似乎并没有发生过拟合现象。

你可以将`validation_split`设置为希望Keras用于验证的训练集的比率，而不是使用`validation_data`参数传递验证集。例如，`validation_split=0.1`告诉Keras使用数据的最后10％（在乱序之前）进行验证。

如果训练集非常不平衡，其中某些类的代表过多，而其他类的代表不足，那么在调用`fit()`方法时设置`class_weight`参数会很有用，这给代表性不足的类更大的权重，给代表过多的类更小的权重。Keras在计算损失时将使用这些权重。如果你需要每个实例的权重，设置`sample_weight`参数（如果`class_weight`和`sample_weight`都提供了，Keras会把它们相乘）。如果某些实例由专家标记，而另一些实例使用众包平台标记，则按实例权重可能会有用：你可能希望为前者赋予更多权重。你还可以通过将其作为`validation_data`元组的第三项添加到验证集中来提供样本权重（但不提供类权重）。

`fit()`方法返回一个History对象，其中包含训练参数（`history.params`）、经历的轮次列表（`history.epoch`），最重要的是包含在训练集和验证集（如果有）上的每个轮次结束时测得的损失和额外指标的字典（`history.history`）。如果使用此字典创建pandas DataFrame并调用其`plot()`方法，则会获得如图15所示的学习曲线：

![fig15_学习曲线](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig15_%E5%AD%A6%E4%B9%A0%E6%9B%B2%E7%BA%BF.jpg)

你可以看到训练期间训练准确率和验证准确率都在稳步提高，而训练损失和验证损失则在下降。好！而且，验证曲线与训练曲线很接近，这意味着没有太多的过拟合。在这种特殊情况下，该模型看起来在验证集上的表现要好于训练开始时在训练集上的表现。但是事实并非如此：确实，验证误差是在每个轮次结束时计算的，而训练误差是使用每个轮次的运行平均值计算的。因此，训练曲线应向左移动半个轮次。如果这样做，你会看到训练和验证曲线在训练开始时几乎完全重叠。

训练集的性能最终会超过验证性能，就像通常情况下训练足够长的时间一样。你可以说模型尚未完全收敛，因为验证损失仍在下降，因此你可能应该继续训练。这就像再次调用`fit()`方法那样简单，因为Keras只是从它停止的地方继续训练（你应该能够达到接近89％的验证准确率）。

如果你对模型的性能不满意，则应回头调整超参数。首先要检查的是学习率。如果这样做没有帮助，请尝试使用另一个优化器（并在更改任何超参数后始终重新调整学习率）。如果性能仍然不佳，则尝试调整模型超参数（例如层数、每层神经元数以及用于每个隐藏层的激活函数的类型）。你还可以尝试调整其他超参数，例如批处理大小（可以使用`batch_size`参数在`fit()`方法中进行设置，默认为32）。在本章的最后，我们将回到超参数调整。对模型的验证精度感到满意后，应在测试集上对其进行评估泛化误差，然后再将模型部署到生产环境中。你可以使用`evaluate()`方法轻松完成此操作（它还支持其他几个参数，例如`batch_size`和`sample_weight`，请查看文档以获取更多详细信息）：


```python
    model.evaluate(X_test, y_test)
    >>> 313/313 [==============================] - 0s 658us/step - loss: 0.3373 - accuracy: 0.8834
    [0.3372957408428192, 0.883400022983551]
```

正如我们在第2章中看到的那样，在测试集上获得比在验证集上略低的性能是很常见的，因为超参数是在验证集而不是测试集上进行调优的（但是在本示例中，我们没有做任何超参数调整，因此较低的精度只是运气不好）。切记不要调整测试集上的超参数，否则你对泛化误差的估计将过于乐观。

#### 使用模型进行预测

接下来，我们可以使用模型的`predict()`方法对新实例进行预测。由于没有实际的新实例，因此我们将仅使用测试集的前三个实例：

```python
    X_new = X_test[:3]
    y_proba = model.predict(X_new)
    y_proba.round(2)

    >>> array
    ([[0.  , 0.  , 0.  , 0.  , 0.  , 0.01, 0.  , 0.03, 0.  , 0.96],
    [0.  , 0.  , 0.99, 0.  , 0.01, 0.  , 0.  , 0.  , 0.  , 0.  ],
    [0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]],
    dtype=float32)
```

如你所见，对于每个实例，模型估计从0类到9类每个类的概率。例如，对于第一个图像，模型估计是第9类（脚踝靴）的概率为96％，第5类的概率（凉鞋）为3％，第7类（运动鞋）的概率为1％，其他类别的概率可忽略不计。换句话说，它“相信”第一个图像是鞋类，最有可能是脚踝靴，但也可能是凉鞋或运动鞋。如果你只关心估计概率最高的类（即使该概率非常低），则可以使用`pre dict_classes()`方法：

```python
    #y_pred = model.predict_classes(X_new) # deprecated
    y_pred = np.argmax(model.predict(X_new), axis=-1)
    y_pred
    >>> array([9, 2, 1], dtype=int64)
    np.array(class_names)[y_pred]
    >>> array(['Ankle boot', 'Pullover', 'Trouser'], dtype='<U11')
```

在这里，分类器实际上对所有三个图像进行了正确分类（图像如图16所示）：

![fig16_正确分类的Fashion MNIST图像](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig16_%E6%AD%A3%E7%A1%AE%E5%88%86%E7%B1%BB%E7%9A%84Fashion%20MNIST%E5%9B%BE%E5%83%8F.jpg)

```python
    y_new = y_test[:3]
    y_new
    >>> array([9, 2, 1], dtype=uint8)
```

现在，你知道如何使用顺序API来构建、训练、评估和使用分类MLP。但是回归呢？

### 10.2.3 用顺序API构建回归MLP

让我们转到加州的住房问题，并使用回归神经网络解决它。为简单起见，我们将使用Scikit-Learn的`fetch_california_housing()`函数加载数据。该数据集比我们在第2章中使用的数据集更简单，因为它仅包含数字特征（没有`ocean_proximity`特征），并且没有缺失值。加载数据后，我们将其分为训练集、验证集和测试集，然后比例缩放所有特征：

```python
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    housing = fetch_california_housing()

    X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
```

使用顺序API来构建、训练、评估和使用回归MLP进行预测与我们进行分类非常相似。主要区别在于输出层只有一个神经元（因为我们只预测一个单值），并且不使用激活函数，而损失函数是均方误差。由于数据集噪声很大，我们只使用比以前少的神经元的单层隐藏层，以避免过拟合：

```python
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
        keras.layers.Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
    mse_test = model.evaluate(X_test, y_test)
    X_new = X_test[:3]
    y_pred = model.predict(X_new)
```

如你所见，顺序API非常易于使用。但是尽管顺序模型非常普遍，但有时构建具有更复杂拓扑结构或具有多个输入或输出的神经网络还是常见的。为此，Keras提供了函数式API。

### 10.2.4 使用函数式API构建复杂模型

非顺序神经网络的一个示例是“宽深”神经网络。这种神经网络架构是由Heng-Tze Cheng等人在2016年发表的论文引入的。它将所有或部分输入直接连接到输出层，如图17所示。这种架构使神经网络能够学习深度模式（使用深度路径）和简单规则（通过短路径）。相比之下，常规的MLP迫使所有数据流经整个层的堆栈。

![fig17_宽深神经网络](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig17_%E5%AE%BD%E6%B7%B1%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.jpg)

因此，数据的简单模式最终可能会因为顺序被转换而失真。

让我们建立这样一个神经网络来解决加州的住房问题：

```python
    input_ = keras.layers.Input(shape=X_train.shape[1:])
    hidden1 = keras.layers.Dense(30, activation="relu")(input_)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.concatenate([input_, hidden2])
    output = keras.layers.Dense(1)(concat)
    model = keras.models.Model(inputs=[input_], outputs=[output])
```

让我们遍历这段代码的每一行：

- 首先，我们需要创建一个Input对象。这是模型需要的输入类型的规范，包括其`shape`和`dtype`。我们很快就会看到，一个模型实际上可能有多个输入。

- 接下来，我们创建一个包含30个神经元的Dense层，使用ReLU激活函数。创建它后，请注意，我们像调用函数一样将其传递给输入。这就是将其称为函数式API的原因。注意，我们只是在告诉Keras它应该如何将各层连接在一起。尚未处理任何实际数据。

- 然后，我们创建第二个隐藏层，然后再次将其用作函数。请注意，我们将第一个隐藏层的输出传递给它。

- 接下来，我们创建一个Concatenate层，再次像函数一样立即使用它来合并输入和第二个隐藏层的输出。你可能更喜欢`keras.layers.concatenate()`函数，该函数创建一个`Concatenate`层并立即使用给定的输入对其进行调用。

- 然后我们创建具有单个神经元且没有激活函数的输出层，然后像函数一样调用它，将合并结果传递给它。

- 最后，我们创建一个Keras Model，指定要使用的输入和输出。

一旦构建了Keras模型，一切都与之前的一样，因此无须在此处重复：你必须编译模型，对其进行训练，评估并使用它来进行预测。

但是如果你想通过宽路径送入特征的子集，而通过深路径送入特征的另一个子集（可能有重合）呢（见图18）？在这种情况下，一种解决方案是使用多个输入。例如，假设我们要通过宽路径送入5个特征（特征0到4），并通过深路径送入6个特征（特征2到7）：

![fig18_处理多输入](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig18_%E5%A4%84%E7%90%86%E5%A4%9A%E8%BE%93%E5%85%A5.jpg)

该代码是不言自明的。你应该至少命名最重要的层，尤其是当模型变得有点复杂时。请注意，在创建模型时，我们指定了`input=[input_A,input_B]`。现在我们可以像往常一样编译模型了，但是当我们调用`fit()`方法时，必须传递一对矩阵`(X_train_A,X_train_B)`：各输入一个矩阵，而不是传递单个输入矩阵`X_train`。当你调用`evaluate()`或`predict()`时，X_valid、X_test和X_new同样如此：

```python
    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
    history = model.fit(X_train, y_train, epochs=20,
                        validation_data=(X_valid, y_valid))
    mse_test = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_new)
```

在许多用例中，你可能需要多个输出：

- 这个任务可能会需要它。例如，你可能想在图片中定位和分类主要物体。这既是回归任务（查找物体中心的坐标以及宽度和高度），又是分类任务。

- 同样，你可能有基于同一数据的多个独立任务。当然你可以为每个任务训练一个神经网络，但是在许多情况下，通过训练每个任务一个输出的单个神经网络会在所有任务上获得更好的结果。这是因为神经网络可以学习数据中对任务有用的特征。例如，你可以对面部图片执行多任务分类，使用一个输出对人的面部表情进行分类（微笑、惊讶等），使用另一个输出来识别他们是否戴着眼镜。

- 另一个示例是作为正则化技术（即训练约束，其目的是减少过拟合，从而提高模型的泛化能力）。例如，你可能希望在神经网络结构中添加一些辅助输出（见图19），以确保网络的主要部分自己能学习有用的东西，而不依赖于网络的其余部分。

![fig19_正则化](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig19_%E6%AD%A3%E5%88%99%E5%8C%96.jpg)

添加额外的输出非常容易：只需将它们连接到适当的层，然后将它们添加到模型的输出列表中即可。例如，以下代码构建了如图19所示的网络：

```python
   input_A = keras.layers.Input(shape=[5], name="wide_input")
    input_B = keras.layers.Input(shape=[6], name="deep_input")
    hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.concatenate([input_A, hidden2])
    output = keras.layers.Dense(1, name="main_output")(concat)
    aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
    model = keras.models.Model(inputs=[input_A, input_B],
                            outputs=[output, aux_output]) 
```

每个输出都需要自己的损失函数。因此当我们编译模型时，应该传递一系列损失（如果传递单个损失，Keras将假定所有输出必须使用相同的损失）。默认情况下，Keras将计算所有这些损失，并将它们简单累加即可得到用于训练的最终损失。我们更关心主要输出而不是辅助输出（因为它仅用于正则化），因此我们要给主要输出的损失更大的权重。幸运的是，可以在编译模型时设置所有的损失权重：

```python
    model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))
```

现在当训练模型时，需要为每个输出提供标签。在此示例中，主要输出和辅助输出应预测出相同的结果，因此它们应使用相同的标签。除了传递`y_train`之外，还需要传递`(y_train, y_train)`（对于`y_valid`和`y_test`也是如此）：

```python
    history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20,
                        validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))
```

当评估模型时，Keras将返回总损失以及所有单个损失，同样，`predict()`方法将为每个输出返回预测值：

```python
    total_loss, main_loss, aux_loss = model.evaluate(
    [X_test_A, X_test_B], [y_test, y_test])
    y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B]) 
```

如你所见，你可以使用函数式API轻松构建所需的任何网络结构。让我们看一下构建Keras模型的最后一种方法。

### 10.2.5 使用子类API构建动态模型

顺序API和函数式API都是声明性的：首先声明要使用的层以及应该如何连接它们，然后才能开始向模型提供一些数据进行训练或推断。这具有许多优点：可以轻松地保存、克隆和共享模型；可以显示和分析它的结构；框架可以推断形状和检查类型，因此可以及早发现错误（即在任何数据通过模型之前）。由于整个模型是一个静态图，因此调试起来也相当容易。但另一方面是它是静态的。一些模型涉及循环、变化的形状、条件分支和其他动态行为。对于这种情况，或者只是你喜欢命令式的编程风格，则子类API非常适合你。

只需对Model类进行子类化，在构造函数中创建所需的层，然后在`call()`方法中执行所需的计算即可。例如，创建以下WideAndDeepModel类的实例将给我们一个等效于刚刚使用函数式API构建的模型。然后，你可以像我们刚做的那样对其进行编译、评估并使用它进行预测：

```python
    class WideAndDeepModel(keras.models.Model):
        def __init__(self, units=30, activation="relu", **kwargs):
            super().__init__(**kwargs)
            self.hidden1 = keras.layers.Dense(units, activation=activation)
            self.hidden2 = keras.layers.Dense(units, activation=activation)
            self.main_output = keras.layers.Dense(1)
            self.aux_output = keras.layers.Dense(1)
            
        def call(self, inputs):
            input_A, input_B = inputs
            hidden1 = self.hidden1(input_B)
            hidden2 = self.hidden2(hidden1)
            concat = keras.layers.concatenate([input_A, hidden2])
            main_output = self.main_output(concat)
            aux_output = self.aux_output(hidden2)
            return main_output, aux_output

    model = WideAndDeepModel(30, activation="relu")
```

这个示例看起来非常类似于函数式API，只是我们不需要创建输入。我们只使用`call()`方法的输入参数，就可以将构造函数中层的创建与其在`call()`方法中的用法分开。最大的区别是你可以在`call()`方法中执行几乎所有你想做的操作：for循环、if语句、底层TensorFlow操作，等等（见第12章）。这使得它成为研究新想法的研究人员的绝佳API。

这种额外的灵活性的确需要付出一定的代价：模型的架构隐藏在`call()`方法中，因此Keras无法对其进行检查。它无法保存或克隆。当你调用`summary()`方法时，你只会得到一个图层列表，而没有有关它们如何相互连接的信息。而且Keras无法提前检查类型和形状，并且更容易出错。因此，除非你确实需要这种额外的灵活性，否则你应该坚持使用顺序API或函数式API。

Keras模型可以像常规层一样使用，因此你可以轻松地将它们组合以构建复杂的结构。

现在你知道如何使用Keras构建和训练神经网络了，那么如何保存它们呢？

### 10.2.6 保存和还原模型

使用顺序API或函数式API时，保存训练好的Keras模型非常简单：

```python
    model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])    
```

Keras使用HDF5格式保存模型的结构（包括每一层的超参数）和每一层的所有模型参数值（例如，连接权重和偏置）。它还可以保存优化器（包括其超参数及其可能具有的任何状态）。在第19章，我们可以看到如何使用Tensorflow的SavedModel格式来保存一个tf.keras模型。

通常你有一个训练模型并保存模型的脚本，以及一个或多个加载模型并使用其进行预测的脚本（或Web服务）。加载模型同样简单：

```python
    model = keras.models.load_model("my_keras_model.h5")
```

当使用顺序API或函数式API时，这是适用的，但不幸的是，在使用模型子类化时，它将不起作用。你至少可以使用`save_weights()`和`load_weights()`来保存和还原模型参数，但是你需要自己保存和还原其他所有内容。

但是，如果训练持续几个小时怎么办？这是很常见的，尤其是在大型数据集上进行训练时。在这种情况下，你不仅应该在训练结束时保存模型，还应该在训练过程中定期保存检查点，以免在计算机崩溃时丢失所有内容。但是如何告诉`fit()`方法保存检查点呢？使用回调。

### 10.2.7 使用回调函数

`fit()`方法接受一个callbacks参数，该参数使你可以指定Keras在训练开始和结束时，每个轮次的开始和结束时（甚至在处理每个批量之前和之后）将调用的对象列表。例如，在训练期间ModelCheckpoint回调会定期保存模型的检查点，默认情况下，在每个轮次结束时：

```python
        model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
    checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb])
    model = keras.models.load_model("my_keras_model.h5") # rollback to best model
    mse_test = model.evaluate(X_test, y_test)
```

此外，如果在训练期间使用验证集，则可以在创建ModelCheckpoint时设置`save_best_only=True`。在这种情况下，只有在验证集上的模型性能达到目前最好时，它才会保存模型。这样，你就不必担心训练时间太长而过拟合训练集：只需还原训练后保存的最后一个模型，这就是验证集中的最佳模型。以下代码是实现提前停止的简单方法（见第4章）：

```python
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
    checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb])
    model = keras.models.load_model("my_keras_model.h5") # rollback to best model
    mse_test = model.evaluate(X_test, y_test)
```

实现提前停止的另一种方法是使用EarlyStopping回调。如果在多个轮次（由`patience`参数定义）的验证集上没有任何进展，它将中断训练，并且可以选择回滚到最佳模型。你可以将两个回调结合起来以保存模型的检查点（以防计算机崩溃），并在没有更多进展时尽早中断训练（以避免浪费时间和资源）：

```python
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                    restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb, early_stopping_cb])
```

可以将轮次数设置为较大的值，因为训练将在没有更多进展时自动停止。在这种情况下，无须还原保存的最佳模型，因为EarlyStopping回调将跟踪最佳权重，并在训练结束时为你还原它。

如果需要额外的控制，则可以轻松编写自己的自定义回调。作为如何执行的示例，以下自定义回调将显示训练过程中验证损失与训练损失之间的比率（例如，检测过拟合）：

```python
    class PrintValTrainRatioCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))
```

现在让我们看看使用tf.keras时肯定应该在工具箱中有的另一种工具：TensorBoard。

### 10.2.8 使用TensorBoard进行可视化

TensorBoard是一款出色的交互式可视化工具，可用于在训练期间查看学习曲线；比较多次运行的学习曲线；可视化计算图；分析训练统计数据；查看由模型生成的图像；把复杂的多维数据投影到3D，自动聚类并进行可视化，等等！安装TensorFlow时会自动安装此工具，因此你已经拥有了它。

要使用它，你必须修改程序以便将要可视化的数据输出到名为事件文件的特殊二进制日志文件中。每个二进制数据记录称为摘要。TensorBoard服务器将监视日志目录，并将自动获取更改并更新可视化效果：这使你可以可视化实时数据（有短暂延迟），例如训练期间的学习曲线。通常你需要把TensorBoard服务器指向根日志目录并配置程序，以使其在每次运行时都写入不同的子目录。这样相同的TensorBoard服务器实例可以使你可视化并比较程序多次运行中的数据，而不会混淆所有内容。

让我们首先定义用于TensorBoard日志的根日志目录，再加上一个将根据当前日期和时间生成一个子目录的函数，以便每次运行时都不同。你可能希望在日志目录中包含其他信息，例如你正在测试的超参数值，以使你更容易知道在TensorBoard中查看的内容：

```python
    import os
    root_logdir = os.path.join(os.curdir, "my_logs")

    def get_run_logdir():
        import time
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)

    run_logdir = get_run_logdir()
    run_logdir
```

好消息是Keras提供了一个不错的`TensorBoard()`回调：

```python
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=[8]),
        keras.layers.Dense(30, activation="relu"),
        keras.layers.Dense(1)
    ])    
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb, tensorboard_cb])
```

这就是全部！不难使用。如果运行此代码，`TensorBoard()`回调将为你创建日志目录（如果需要，还有父目录），并且在训练期间它将创建事件文件并向其写入摘要。第二次运行该程序（也许更改某些超参数值）后，你将得到一个类似于以下内容的目录结构：

```python

```

每次运行有一个目录，每个目录包含一个用于训练日志的子目录和一个用于验证日志的子目录。两者都包含事件文件，但是训练日志还包含概要分析跟踪：这使TensorBoard可以准确显示模型在所有设备上花费在模型各部分上的时间，对于查找性能瓶颈非常有用。

接下来，你需要启动TensorBoard服务器。一种方法是在终端中运行命令。如果你在虚拟环境中安装了TensorFlow，则应将其激活。接下来在项目的根目录（或从任何其他位置，只要指向适当的日志目录）运行以下命令：

```python
    $ tensorboard --logdir=./my_logs --port=6006
    TensorBoard 2.0.0 at http://mycomputer.local:6006/ (Press CTRL+C to quit)
```

如果你的终端找不到tensorboard脚本，那你必须更新PATH环境变量，以便它包含脚本安装目录（或者你可以在命令行中用python3-m tensorboard.main替换tensorboard）。服务器启动后，你可以打开Web浏览器并转到http://localhost:6006。

或者你可以直接在Jupyter中运行以下命令来使用TensorBoard。第一行加载TensorBoard扩展，第二行在端口6006上启动TensorBoard服务器（除非它已经启动）并连接到它：

```python
    %load_ext tensorboard
    %tensorboard --logdir=./my_logs --port=6006
```

无论用哪种方式，你都应该看到TensorBoard的Web界面。单击“SCALARS”选项卡以查看学习曲线（见图10-17）。在左下方，选择要显示的日志（例如，第一次和第二次运行的训练日志），然后单击epoch_loss标量。请注意两次运行的训练损失都下降得很好，但是第二次下降得更快。实际上，我们使用的学习率为0.05（optimizer=keras.optimizers.SGD（lr=0.05）），而不是0.001。

你还可以可视化整个图形、学习的权重（投影到3D）或分析跟踪。`TensorBoard()`回调也具有记录额外数据的选项，例如嵌入：

![fig20_用TensorBoard可视化学习曲线](https://github.com/Vuean/Hands-On-ML/blob/main/Chapter10/figures/fig20_%E7%94%A8TensorBoard%E5%8F%AF%E8%A7%86%E5%8C%96%E5%AD%A6%E4%B9%A0%E6%9B%B2%E7%BA%BF.jpg)

此外，TensorFlow在tf.summary包中提供了一个较底层的API。以下代码使用`create_file_writer()`函数创建一个S ummaryWriter，并将该函数用作上下文来记录标量、直方图、图像、音频和文本，然后可以使用TensorBoard将其可视化（尝试一下！）

让我们总结一下你到目前为止在本章中学到的知识：我们了解了神经网络的来源，MLP是什么以及如何将其用于分类和回归，如何使用tf.keras的顺序API来构建MLP，以及如何使用函数式API或子类API来构建更复杂的模型架构。你学习了如何保存和还原模型，以及如何使用回调函数来保存检查点，提前停止，等等。最后，你学习了如何使用TensorBoard进行可视化。你已经可以使用神经网络来解决许多问题了！但是，你可能想知道如何选择隐藏层的数量、网络中神经元的数量以及所有其他超参数。让我们现在来看一下。

## 10.3 微调神经网络超参数

神经网络的灵活性也是它们的主要缺点之一：有许多需要调整的超参数。你不仅可以使用任何可以想象的网络结构，而且即使在简单的MLP中，你也可以更改层数、每层神经元数、每层要使用的激活函数的类型、权重初始化逻辑，以及更多。你如何知道哪种超参数最适合你的任务？

一种选择是简单地尝试超参数的许多组合，然后查看哪种对验证集最有效（或使用K折交叉验证）。例如我们可以像第2章中一样使用GridSearchCV或RandomizedSearchCV来探索超参数空间。为此我们需要将Keras模型包装在模仿常规Scikit-Learn回归器的对象中。第一步是创建一个函数，该函数将在给定一组超参数的情况下构建并编译Keras模型：

```python
    def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=input_shape))
        for layer in range(n_hidden):
            model.add(keras.layers.Dense(n_neurons, activation="relu"))
        model.add(keras.layers.Dense(1))
        optimizer = keras.optimizers.SGD(lr=learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
        return model
```

函数为单变量回归（仅一个输出神经元）创建简单的Sequential模型，使用给定的输入形状以及给定数量的隐藏层和神经元，并使用配置了指定学习率的SGD优化器对其进行编译。像Scikit-Learn一样，最好为尽可能多的超参数提供合理的默认值。

接下来，让我们基于`build_model()`函数创建一个KerasRegressor：

```python
    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
```

KerasRegressor对象是使用`build_model()`构建的Keras模型的一个包装。由于创建时未指定任何超参数，因此它将使用我们在`build_model()`中定义的默认超参数。现在，我们可以像常规Scikit-Learn回归器一样使用该对象：我们可以使用其`fit()`方法进行训练，然后使用其`score()`方法进行评估，然后使用其`predict()`方法进行预测，如以下代码所示：

```python
    keras_reg.fit(X_train, y_train, epochs=100,
                validation_data=(X_valid, y_valid),
                callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    mse_test = keras_reg.score(X_test, y_test)
    y_pred = keras_reg.predict(X_new)
```

请注意，传递给`fit()`方法的任何其他参数都将传递给内部的Keras模型。还要注意，该分数将与MSE相反，因为Scikit-Learn希望获得分数，而不是损失（即分数越高越好）。

我们不想训练和评估这样的单个模型，尽管我们想训练数百个变体，并查看哪种变体在验证集上表现最佳。由于存在许多超参数，因此最好使用随机搜索而不是网格搜索（见第2章）。让我们尝试探索隐藏层的数量、神经元的数量和学习率：

```python
    from scipy.stats import reciprocal
    from sklearn.model_selection import RandomizedSearchCV

    param_distribs = {
        "n_hidden": [0, 1, 2, 3],
        "n_neurons": np.arange(1, 100)               .tolist(),
        "learning_rate": reciprocal(3e-4, 3e-2)      .rvs(1000).tolist(),
    }

    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
    rnd_search_cv.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[keras.callbacks.EarlyStopping(patience=10)])
```

这与我们在第2章中所做的相同，只是这里我们将额外的参数传递给`fit()`方法，并将它们传递给内部的Keras模型。请注意，RandomizedSearchCV使用K折交叉验证，因此它不使用X_valid和y_valid，它们仅用于提前停止。

探索可能持续数小时，具体时间取决于硬件、数据集的大小、模型的复杂性以及n_iter和cv的值。当结束时，你可以访问找到的最佳参数、最佳分数和经过训练的Keras模型，如下所示：

```python
    rnd_search_cv.best_params_
    >>> 
    rnd_search_cv.best_score_
    >>> 
    rnd_search_cv.best_estimator_
```

现在你可以保存该模型，在测试集上对其进行评估，如果对它的性能满意，可以将其部署到生产环境中。使用随机搜索并不困难，它可以很好地解决许多相当简单的问题。但是，当训练速度很慢时（对于具有较大数据集的更复杂的问题），此方法将仅探索超参数空间的一小部分。你可以通过手动协助搜索过程来部分缓解此问题：首先使用宽范围的超参数值快速进行随机搜索，然后使用以第一次运行中找到的最佳值为中心，用较小范围的值运行另一个搜索，以此类推。该方法有望放大一组好的超参数。但是，这非常耗时，也不能最好地利用时间。

幸运的是，有许多技术可以比随机方法更有效地探索搜索空间。它们的核心思想很简单：当空间的某个区域被证明是好的时，应该对其进行更多的探索。此类技术可为你解决“缩放”过程，并在更短的时间内提供更好的解决方案。以下是一些可用于优化超参数的Python库：

Hyperopt

一个流行的库，用于优化各种复杂的搜索空间（包括诸如学习率的实数值和诸如层数的离散值）。

Hyperas、kopt或Talos

有用的库，用于优化Keras模型的超参数（前两个基于Hyperopt）。

Keras Tuner

Google针对Keras模型提供的易于使用的超参数优化库，可用于可视化和分析的托管服务。

Scikit-Optimize（skopt）

通用优化库。BayesSearchCV类使用类似于GridSearchCV的接口来进行贝叶斯优化。

Spearmint

贝叶斯优化库。

Hyperband

基于Lisha Li等人的最新Hyperband论文的快速超参数调整库。

Sklearn-Deap

基于进化算法的超参数优化库，具有类似于GridSearchCV的接口。

此外许多公司提供超参数优化服务。我们将在第19章中讨论Google Cloud AI Platform的超参数调整服务。其他选项包括Arimo和SigOpt的服务以及CallDesk的Oscar。

超参数调整仍然是研究的活跃领域，而进化算法正在卷土重来。例如，查看DeepMind的2017年的优秀论文，作者同时优化了整体模型及其超参数。Google还使用了一种进化方法，不仅用于搜索超参数，而且还为该问题寻找最佳神经网络架构。他们的AutoML套件已作为云服务提供。也许手动构建神经网络的时代即将结束？可以查看Google关于该主题的文章。实际上，进化算法已成功用于训练单个神经网络，从而取代了无处不在的梯度下降！请参阅Uber在2017年发表的文章，作者介绍了他们的Deep Neuroevolution技术。

但是尽管取得了这些令人振奋的进步以及这些工具和服务，但是了解每个超参数的合理值，仍然有助于你构建快速原型并限制搜索空间。以下各节为选择MLP中隐藏层和神经元的数量以及为某些主要超参数选择合适的值提供了指导。

### 10.3.1 隐藏层数量

对于许多问题，你可以从单个隐藏层开始并获得合理的结果。只要具有足够的神经元，只有一个隐藏层的MLP理论上就可以对最复杂的功能进行建模。但是对于复杂的问题，深层网络的参数效率要比浅层网络高得多：与浅层网络相比，深层网络可以使用更少的神经元对复杂的函数进行建模，从而使它们在相同数量的训练数据下可以获得更好的性能。

为了理解原因，假设要求你使用某些绘图软件来绘制森林，但禁止复制和粘贴任何内容。这要花费大量的时间：你必须分别绘制每棵树（逐个分支，逐个叶子）。如果你可以改为绘制一片叶子，将其复制并粘贴以绘制一个分支，然后复制并粘贴该分支以创建一棵树，最后复制并粘贴此树以创建森林，那么你将很快完成。现实世界中的数据通常以这种层次结构进行构造，而深度神经网络会自动利用这一事实：较低的隐藏层对低层结构（例如形状和方向不同的线段）建模，中间的隐藏层组合这些低层结构，对中间层结构（例如正方形、圆形）进行建模，而最高的隐藏层和输出层将这些中间结构组合起来，对高层结构（例如人脸）进行建模。

这种分层架构不仅可以帮助DNN更快地收敛到一个好的解，而且还可以提高DNN泛化到新数据集的能力。例如，如果你已经训练了一个模型来识别图片中的人脸，并且现在想训练一个新的神经网络来识别发型，则可以通过重用第一个网络的较低层来开始训练。你可以将它们初始化为第一个网络较低层的权重和偏置值，而不是随机初始化新神经网络前几层的权重和偏置值。这样，网络就不必从头开始学习大多数图片中出现的所有底层结构。只需学习更高层次的结构（例如发型）。这称为迁移学习。

总而言之，对于许多问题，你可以仅从一两个隐藏层开始，然后神经网络就可以正常工作。例如，仅使用一个具有几百个神经元的隐藏层，就可以轻松地在MNIST数据集上达到97％以上的准确率，而使用具有相同总数的神经元的两个隐藏层，可以在大致相同训练时间上轻松达到98％以上的精度。对于更复杂的问题，你可以增加隐藏层的数量，直到开始过拟合训练集为止。非常复杂的任务（例如图像分类或语音识别）通常需要具有数十层（甚至数百层，但不是全连接的网络，如我们将在第14章中看到的）的网络，并且它们需要大量的训练数据。你几乎不必从头开始训练这样的网络：重用一部分类似任务的经过预训练的最新网络更为普遍。这样，训练就会快得多，所需的数据也要少得多（我们将在第11章中对此进行讨论）。

### 10.3.2 每个隐藏层的神经元数量

输入层和输出层中神经元的数量取决于任务所需的输入类型和输出类型。例如，MNIST任务需要28×28=784个输入神经元和10个输出神经元。

对于隐藏层，通常将它们调整大小以形成金字塔状，每一层的神经元越来越少，理由是许多低层特征可以合并成更少的高层特征。MNIST的典型神经网络可能具有3个隐藏层，第一层包含300个神经元，第二层包含200个神经元，第三层包含100个神经元。但是，这种做法已被很大程度上放弃了，因为似乎在所有隐藏层中使用相同数量的神经元，在大多数情况下层的表现都一样好，甚至更好；另外，只需要调整一个超参数，而不是每层一个。也就是说，根据数据集，有时使第一个隐藏层大于其他隐藏层是有帮助的。

就像层数一样，你可以尝试逐渐增加神经元的数量，直到网络开始过拟合为止。但是在实践中，选择一个比你实际需要的层和神经元更多的模型，然后使用提前停止和其他正则化技术来防止模型过拟合，通常更简单、更有效。Google的科学家Vincent Vanhoucke称之为“弹力裤”方法：与其浪费时间寻找与自己的尺码完全匹配的裤子，不如使用大尺寸的弹力裤来缩小到合适的尺寸。使用这种方法，一方面，可以避免可能会破坏模型的瓶颈层。另一方面，如果一层的神经元太少，它将没有足够的表征能力来保留来自输入的所有有用信息（例如具有两个神经元的层只能输出2D数据，因此如果它处理3D数据，一些信息将会丢失）。无论网络的其余部分有多强大，这些信息都将永远无法恢复。

通常通过增加层数而不是每层神经元数，你将获得更多收益。

### 10.3.3 学习率、批量大小和其他超参数

隐藏层和神经元的数量并不是你可以在MLP中进行调整的唯一超参数。以下是一些最重要的信息，以及有关如何设置它们的提示：

学习率

学习率可以说是最重要的超参数。一般而言，最佳学习率约为最大学习率的一半（即学习率大于算法发散的学习率，如我们在第4章中看到的）。找到一个好的学习率的一种方法是对模型进行数百次迭代训练，从非常低的学习率（例如10-5）开始，然后逐渐将其增加到非常大的值（例如10）。这是通过在每次迭代中将学习率乘以恒定因子来完成的（例如，将exp（log（106）/500）乘以500次迭代中的10-5到10）。如果将损失作为学习率的函数进行绘制（对学习率使用对数坐标），你应该首先看到它在下降。但是过一会儿学习率将过大，因此损失将重新上升：最佳学习率将比损失开始攀升的点低一些（通常比转折点低约10倍）。然后你可以重新初始化模型，并以这种良好的学习率正常训练模型。我们将在第11章中介绍更多的学习率技术。

优化器

选择比普通的小批量梯度下降更好的优化器（并调整其超参数）也很重要。我们将在第11章中了解几个高级优化器。

批量大小

批量的大小可能会对模型的性能和训练时间产生重大影响。使用大批量的主要好处是像GPU这样的硬件加速器可以有效地对其进行处理（见第19章），因此训练算法每秒会看到更多的实例。因此，许多研究人员和从业人员建议使用可容纳在GPU RAM中的最大批量。但是这里有一个陷阱：在实践中，大批量通常会导致训练不稳定，尤其是在训练开始时，结果模型泛化能力可能不如小批量训练的模型。2018年4月，Yann LeCun甚至在推特上写道：“朋友不要让朋友使用大于32的小批量处理。”并引用了Dominic Masters和CarloLuschi在2018年发表的一篇论文，得出的结论是首选使用小批量（从2到32），因为小批量可以在更少的训练时间内获得更好的模型。但是，其他论文则提出相反意见。在2017年，Elad Hoffer等人和Priya Goyal等人的论文表明，可以通过各种技术手段使用非常大的批量处理（最多8192），例如提高学习率（即开始学习以较低的学习率进行训练，然后提高学习率，如第11章所述）。这导致了非常短的训练时间，没有泛化能力的差距。因此一种策略是尝试使用大批量处理，慢慢增加学习率，如果训练不稳定或最终表现令人失望，则尝试使用小批量处理。

激活函数

我们在本章前面讨论了如何选择激活函数：通常，ReLU激活函数是所有隐藏层的良好的默认设置。对于输出层，这实际上取决于你的任务。

迭代次数

在大多数情况下，实际上不需要调整训练迭代次数，只需使用提前停止即可。

最佳学习率取决于其他超参数，尤其是批量大小，因此如果你修改了任何超参数，请确保也更新学习率。

有关调整神经网络超参数的更多最佳实践，请查看Leslie Smith的2018年优秀论文。到此结束我们对人工神经网络及其在Keras中实现的介绍。在接下来的几章中，我们将讨论训练非常深度网络的技术。我们还将探索如何使用TensorFlow的较底层API自定义模型，以及如何使用Data API有效地加载和预处理数据。我们还将深入探讨其他流行的神经网络架构：用于图像处理的卷积神经网络、用于顺序数据的递归神经网络、用于表征学习的自动编码器以及用于建模和生成数据的生成式对抗网络。