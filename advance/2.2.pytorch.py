'''
PyTorch的tensor在概念上与Numpy的array相同，tensor是一个N维数组，PyTorch提供了许多函数用于操作这些张量。
区别在于PyTorch可以利用GPU加速其数值运算。
'''

import torch
dtype = torch.float
device = torch.device('cpu')
# N是批量大小，D_in是输入维度，H是隐藏的维度，D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10
# 创建随机输入和输出数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
# 随机初始化权重
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # 向前传递，计算预测值y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    # 计算和打印损失loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)
    # 反向传播，计算w1和w2对loss的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.T)
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.T.mm(grad_h)
    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2