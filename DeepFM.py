import torch
from torch import nn
import torch.optim as optim
import numpy as np


class DeepFM(nn.Module):
    
    def __init__(self, field_size, feature_sizes, embedding_size=4, hidden_dims=[32, 32], num_classes=1, dropout=[0.5, 0.5], use_cuda=False):
        super(DeepFM, self).__init__()

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size

        # Embedding layer: 将输入转化为稠密向量
        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=feature_size, embedding_dim=embedding_size) for feature_size in feature_sizes])
        
        # FM layer: Modeling the interactions between any two features.
        self.fm_first_order_embeds = nn.ModuleList([nn.Embedding(num_embeddings=feature_size, embedding_dim=1) for feature_size in feature_sizes])
        
        # DNN layer: Learning high-order interactions
        all_dims = [field_size * embedding_size] + hidden_dims
        self.dnn = nn.Sequential(
            nn.Dropout(dropout[0]),
            nn.Linear(all_dims[0], all_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(all_dims[1]),
            nn.Dropout(dropout[1]),
            nn.Linear(all_dims[1], all_dims[2]),
            nn.ReLU(),
            nn.BatchNorm1d(all_dims[2]),
        )
        # Output layer: Predicting the final result
        self.fc = nn.Linear(all_dims[-1], num_classes)

    def forward(self, x):
        # 输入数据类型转换
        x = x.long()

        # Embedding layer: 获取 embedding 向量
        embed_x = [self.embeddings[i](x[:, i]) for i in range(self.field_size)]

        # FM first-order term: 获取一阶交互项
        fm_first_order = [self.fm_first_order_embeds[i](x[:, i]) for i in range(self.field_size)]
        fm_first_order = torch.stack(fm_first_order).sum(dim=0)

        # FM second-order term: 获取二阶交互项
        embed_x_sum_square = sum(embed_x) ** 2     # (Σv)^2
        embed_x_square_sum = sum([item**2 for item in embed_x])  # Σv^2
        fm_second_order = 0.5 * (embed_x_sum_square - embed_x_square_sum).sum(1, keepdims=True)

        # DNN layer: 将各个 embedding 向量拼接并输入至 DNN
        dnn_input = torch.flatten(torch.stack(embed_x, dim=-1), start_dim=1)
        dnn_output = self.dnn(dnn_input)

        # Output layer: 综合 FM 和 DNN 预测结果
        out = self.fc(dnn_output) + fm_first_order + fm_second_order

        return torch.sigmoid(out)

if __name__ == "__main__":
    # 配置参数及场景模拟数据
    field_size = 10
    feature_sizes = [1000, 1000, 80, 2, 20, 50, 90, 300, 500, 5000]
    num_examples = 1000000

    epochs = 100
    learning_rate = 0.001
    embedding_size = 4
    hidden_dims = [32, 32]
    num_classes = 1
    dropout = [0.3, 0.3]

    # 随机生成用户行为数据，并转换为Tensor
    X = torch.tensor([np.random.randint(0, size, size=(num_examples, 1)) for size in feature_sizes], dtype=torch.long).squeeze().T

    # 随机生成用户行为标签，并转换为Tensor
    y = torch.tensor(np.random.randint(0, 2, size=(num_examples, 1))).float()

    # 初始化DeepFM模型
    deepfm_model = DeepFM(field_size, feature_sizes, embedding_size, hidden_dims, num_classes, dropout)

    # 设定优化器及损失函数，这里采用二元交叉熵(BCE Loss)
    optimizer = optim.Adam(deepfm_model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # 训练模型
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 正向传播
        outputs = deepfm_model(X)

        loss = criterion(outputs, y)
        #反向传播
        loss.backward()

        #更新参数
        optimizer.step()

        print('Epoch[%d/%d] Loss: %.5f' % (epoch+1, epochs, loss.item()))

    # 测试模型
    test_X = torch.tensor([np.random.randint(0, size, size=(100, 1)) for size in feature_sizes], dtype=torch.long).squeeze().T
    predictions = deepfm_model(test_X)
    print(predictions)