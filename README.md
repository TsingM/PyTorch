# PyTorch

参考：[PyTorch官方教程中文版](www.pytorch123.com)

## 1.开始 tensor.py

### 环境

MacOS + Anaconda(Python3.7) + PyTorch + VSCode

### 简介

Torch是一个有大量机器学习算法支持的科学计算框架，是一个与Numpy类似的张量（Tensor） 操作库，其特点是特别灵活，但因其采用了小众的编程语言是Lua，所以流行度不高，这也就有了PyTorch的出现。所以其实Torch是 PyTorch的前身，它们的底层语言相同，只是使用了不同的上层包装语言。
PyTorch是一个基于Torch的Python开源机器学习库，用于自然语言处理等应用程序。它主要由Facebookd的人工智能小组开发。

### 特点

* 支持GPU

* 灵活，支持动态神经网络

* 底层代码易于理解

* 命令式体验

* 自定义扩展

## 2.自动微分 autograd.py

autograd 包是 PyTorch 中所有神经网络的核心。该包为 Tensors 上的所有操作提供自动微分。它是一个由运行定义的框架，这意味着以代码运行方式定义你的后向传播，并且每次迭代都可以不同。

## 3.神经网络 nn.py

### 神经网络的训练步骤

1. 定义一个神经网络

2. 处理输入，调用反向传播

3. 计算损失

4. 更新参数

### 神经网络的常见类

* torch.Tensor - A multi-dimensional array with support for autograd operations like backward(). Also holds the gradient w.r.t. the tensor.
支持自动微分操作（反向传播等）的多维数组

* nn.Module - Neural network module. Convenient way of encapsulating parameters, with helpers for moving them to GPU, exporting, loading, etc.
神经网络模块，封装参数用于GPU、导出、加载等

* nn.Parameter - A kind of Tensor, that is automatically registered as a parameter when assigned as an attribute to a Module.
一种张量，当作为属性分配给模块时，自动注册为一个参数

* autograd.Function - Implements forward and backward definitions of an autograd operation. Every Tensor operation, creates at least a single Function node, that connects to functions that created a Tensor and encodes its history.
实现自动微分操作的定义，每个张量操作至少生成一个节点

## 4.图像分类器

CIFAR10：包含10个类别的数据集，图像尺寸为33232，即RGB的3层颜色通道，每个通道内尺寸为32*32

训练步骤：

1. 使用torchvision加载并归一化（训练&测试）数据集。

2. 定义卷积神经网络

3. 定义损失函数

4. 在训练集上训练网络

5. 在测试集上测试网络

## 5.数据并行处理

## 6.数据加载和处理

## 7.PyTorch小试牛刀

### PyTorch的两个核心

1. 一个N维张量，类似numpy，但可以在GPU上运行

2. 搭建和训练神经网络时的自动微分/求导机制
