import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# 加载数据
data = pd.read_csv('./免试题2/train.csv', header=None)
X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)

# 定义模型
net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))

# 定义损失函数和优化器
loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.03)

# 训练模型
epochs = 7000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = net(X)
    l = loss(outputs, y)
    l.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {l.item():.4f}')
# 返回模型参数
print(net.state_dict())