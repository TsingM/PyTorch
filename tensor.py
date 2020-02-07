import torch
## 版本查看
print(torch.__version__)

## 构造tensor
# x = torch.rand(5, 3)
# print(x)
# x = torch.zeros(5, 3, dtype = torch.long)
# print(x)
# x = torch.tensor([5.5, 3])
# print(x)

## 基于已有tensor构造tensor
# x = x.new_ones(5, 3, dtype = torch.double)
# print(x)
# x = torch.randn_like(x, dtype = torch.float)
# print(x)
# print(x.size())

## 加法
# y = torch.rand(5, 3)
# print(x + y)
# print(torch.add(x, y))
# result = torch.empty(5, 3)
# torch.add(x, y, out = result)
# print(result)
# y.add_(x)
# print(y)

## 改变tensor
# print(x[:, 1])
# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(-1, 8)
# print(x.size(), y.size(), z.size())
x = torch.randn(1)
print(x)
print(x.item())