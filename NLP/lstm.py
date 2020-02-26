'''
1.序列模型，即输入依赖于时间信息的模型：隐马尔科夫模型HMM、条件随机场CRF。
2.循环神经网络RNN是指可以保持某种状态的神经网络。
3.PyTorch中LSTM的输入形式是一个3D的Tensor。
4.通过out可以取得任意时刻的隐藏状态H。
5.hidden可以取得最后一个时刻的隐藏状态H，用来进行序列的反向传播运算，即将他作为参数传入后面的LSTM网络。
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

lstm = nn.LSTM(3, 3) # 输入维度为3，输出维度为3
inputs = [torch.randn(1, 3) for _ in range(5)] # 生成一个长度为5的序列

# 初始化隐藏状态H
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
# 将序列中的元素逐个输入到LSTM，经过每步操作hidden的值包含了隐藏状态的信息
for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# 额外增加第二个维度
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3)) # 清空隐藏状态H
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)