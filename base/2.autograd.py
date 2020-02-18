import torch

## 创建一个跟踪计算的张量，操作后计算均值的梯度（相当于求1/4 * 3(x+2)^2的导数）
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean() #求均值
print(z, out)
out.backward()
print(x.grad)

## 打开requires_grad标记：requires_grad_
a = torch.randn(2,2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

## 雅克比向量积
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v) #传递向量作参数
print(x.grad)

## 关闭requires_grad标记：no_grad
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)