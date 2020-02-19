import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

# 利用给定数据创建一个torch.tensor对象，这是一个一维向量
V_data = [1., 2., 3.]
V = torch.Tensor(V_data)
print(V)
# 创建一个矩阵
M_data = [[1., 2., 3.], [4., 5., 6.]]
M = torch.Tensor(M_data)
print(M)
# 创建一个2*2*2形式的三维张量
T_data = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
T = torch.Tensor(T_data)
print(T)

# 索引V得到一个标量（0维张量）
print(V[0])
# 从向量V中获取一个数字
print(V[0].item())
# 索引M得到一个向量
print(M[0])
# 索引T得到一个矩阵
print(T[0])
# 创建随机张量 .randn
x = torch.randn((3, 4, 5))
print(x)

# 张量相加
x = torch.Tensor([1., 2., 3.])
y = torch.Tensor([4., 5., 6.])
print(x + y)
# 张量连接 .cat
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)

# 张量重构 .view
x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12))
print(x.view(2, -1))

# 张量标记 .requires_grad
x = torch.randn(3, requires_grad=True)
y = torch.randn(3, requires_grad=True)
z = x + y
print(z)
print(z.grad_fn)
s = z.sum()
print(s.grad_fn)
# 反向 .backward
s.backward()
print(x.grad)

# 标记默认是关闭的
x = torch.randn((2, 2))
y = torch.randn((2, 2))
print(x.requires_grad, y.requires_grad)
z = x + y
print(z.grad_fn)
# 打开标记 .requires_grad_
x = x.requires_grad_()
y = y.requires_grad_()
z = x + y
print(z.grad_fn)
print(z.requires_grad)
# 从z中分离出来但没有历史信息 .detach
new_z = z.detach()
print(new_z.grad_fn)
# 停止跟踪自动求导的标记 .no_grad
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)