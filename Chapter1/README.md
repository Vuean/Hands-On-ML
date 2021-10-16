# 第一章 机器学习概览
  
  过去机器学习的主要应用包括有光学字符识别(Optical Character Recognition, OCR)、垃圾邮件过滤器(spam filter)。

  学习机器学习之前，常常会听到（了解到）监督学习和无监督学习、在线学习和批量学习、基于实例学习和基于模型学习。

  ## 1.1 什么是机器学习

机器学习是一个研究领域，让计算机无须进行明确编程就具备学习能力。从 工程化角度，机器学习是：一个计算机程序利用经验E来学习任务T，性能是P，如果针对任务T的性能P随着经验E不断增长，则称为机器学习。

**训练集**：系统用来进行学习的样例。每个训练样例称作**训练实例**（或**样本**）

## 1.2 为什么使用机器学习

**数据挖掘**：使用机器学习方法挖掘大量数据来帮助发现不太明显的规律，称为数据挖掘。

机器学习适用于：

- 有解决方案但解决方案需要进行大量人工微调或需要遵循大量规则的问题：机器学习算法通常可以简化代码，相比传统方法有更好的性能。

- 传统方法难以解决的复杂问题：最好的机器学习技术也许可以找到解决方案。

- 环境有波动：机器学习算法可以适应新数据。

- 洞察复杂问题和大量数据。

## 1.3 机器学习的应用实例

分析生产线上的产品图像来对产品进行自动分类->图像分类问题->使用卷积神经网络（CNN）；
通过脑部扫描发现肿瘤->语义分割（图像中的每个像素都需要被分类）->使用卷积神经网络（CNN）；
新闻自动分类->自然语言处理（NLP），更具体地是文本分类->可以使用循环神经网络（RNN）、CNN或者Transformer；

## 1.4 机器学习系统的类型

根据是否是在监督下训练可分为：**监督学习**、**无监督学习**、**半监督学习**和**强化学习**。

根据是否能动态地进行增量学习可分为：**在线学习**和**批量学习**。

是简单地将新的数据点和已知的数据点进行匹配，还是对训练数进行模式检测然后建立一个预测模型可分为：**基于实例的学习**和**基于模型的学习**。

### 1.4.1 监督学习和无监督学习

根据训练期间接受的监督数量和监督类型，可以将机器学习系统分为以下四个主要类型：**有监督学习、无监督学习、半监督学习和强化学习**。

1. 监督学习

	在监督学习中，提供给算法的包含所需解决方案的训练集称为**标签**。**分类问题**（classfication problem）是典型的监督学习任务。

	[]()