'''
使用全连接的ReLU网络作为运行示例。该网络将有一个单一的隐藏层，并将使用梯度下降训练，通过最小化网络输出和真正结果的欧几里得距离，来拟合随机生成的数据。
NumPy是用于科学计算的通用框架，它对计算图、深度学习和梯度一无所知，却可以用来手动实现NN。
'''
# -*- coding: utf-8 -*-
import numpy as np
# N是批量大小，D_in是输入维度，H是隐藏的维度，D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10
# 创建随机输入和输出数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)
# 随机初始化权重
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # 向前传递，计算预测值y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    # 计算和打印损失loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)
    # 反向传播，计算w1和w2对loss的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2