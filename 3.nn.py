import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # Net的初始化函数，定义神经网络的组成
    def __init__(self):
        # 复制并使用Net父类的初始化方法，即先运行nn.Module的初始化函数
        super(Net, self).__init__()
        # 定义卷积层：输入维度（初始图像/6张特征图），输出维度（6/16张特征图），卷积核（5*5）
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义全连接层：线性函数（输入维度，输出维度）
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    # 向前传播函数，定义神经网络的结构，反向传播函数基于此自动生成（autograd）
    def forward(self, x):
        # 1.x经过卷积conv1和激活函数ReLU后，使用2*2的窗口进行最大池化，更新
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 2.x经过卷积conv2和激活函数ReLU后，使用2*2的窗口进行最大池化，更新
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 3.x经过view()由张量转换成一维向量（调用num_flat_features）
        x = x.view(-1, self.num_flat_features(x))
        # 4.x经过全连接1和激活函数ReLU，更新
        x = F.relu(self.fc1(x))
        # 5.x经过全连接2和激活函数ReLU，更新
        x = F.relu(self.fc2(x))
        # 6.x经过全连接3，更新
        x = self.fc3(x)
        return x

    # 计算张量x总特征量
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# 查看参数数量与conv1的尺寸
params = list(net.parameters())
print(len(params))
print(params[0].size())

# 随机生成32*32的输入
input = torch.randn(1, 1, 32, 32)
# out = net(input)
# print(out)

# 所有参数梯度缓存置零，用随机梯度进行反向传播
# net.zero_grad()
# out.backward(torch.randn(1, 10))

# 计算与随机目标的均方误差（损失函数）
output = net(input)
target = torch.rand(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)
# 计算路径：input-conv2d-relu-maxpool2d-conv2d-relu-maxpool2d-view-linear-relu-linear-relu-linear-MSELoss-loss
print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU

# 所有参数梯度缓存置零，反向传播损失到各参数的梯度，对比conv1前后变化
net.zero_grad()
print('before: ', net.conv1.bias.grad)
loss.backward()
print('after: ', net.conv1.bias.grad)

# 更新参数（随机梯度下降）
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# 其他更新方法（SGD,Nesterov-SGD,Adam,RMSProp）
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr = 0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()