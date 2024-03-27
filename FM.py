import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

# 1. 定义FM模型
class FactorizationMachine(nn.Module):
    def __init__(self, n, k):
        super(FactorizationMachine, self).__init__()
        self.n = n  # 特征数量
        self.k = k  # 嵌入维度
        self.linear = nn.Linear(n, 1, bias=True)
        self.V = nn.Parameter(torch.randn(n, k))

    def forward(self, x):
        linear_part = self.linear(x)
        inter_part1 = torch.pow(torch.mm(x, self.V), 2)
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))
        interaction = 0.5 * torch.sum(inter_part1 - inter_part2, dim=1, keepdim=True)
        prediction = linear_part + interaction
        return torch.sigmoid(prediction)  # 使用sigmoid函数来输出概率

# 2. 创建一个自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. 生成数据
X, y = make_classification(n_samples=10000, n_features=200, n_informative=2, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 特征归一化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. 将数据包装成DataLoader
train_dataset = CustomDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
test_dataset = CustomDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 6. 初始化FM模型
n_features = X_train.shape[1]
k_factors = 5  # 嵌入维度可以根据需要调整
model = FactorizationMachine(n_features, k_factors)

# 7. 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 8. 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for i, (features, target) in enumerate(train_loader):
        y_pred = model(features).squeeze()
        loss = criterion(y_pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 9. 评估模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    correct = 0
    total = 0
    for features, target in test_loader:
        y_pred = model(features).squeeze()
        predicted = (y_pred >= 0.5).float()  # 将概率大于0.5的视为类别1
        total += target.size(0)
        correct += (predicted == target).sum().item()
    print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')
