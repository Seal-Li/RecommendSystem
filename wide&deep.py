import torch
import torch.nn as nn
import torch.nn.functional as F

# Wide & Deep模型定义
class WideAndDeepModel(nn.Module):
    def __init__(self, wide_dim, deep_dim, hidden_layers):
        super(WideAndDeepModel, self).__init__()
        self.wide = nn.Linear(wide_dim, 1)
        
        # 假设deep_dim是Deep部分输入层的大小
        # 创建Deep部分的层级列表，从deep_dim开始
        deep_layers = [nn.Linear(deep_dim, hidden_layers[0]), nn.ReLU(), nn.Dropout(0.5)]
        for i in range(1, len(hidden_layers)):
            deep_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            deep_layers.append(nn.ReLU())
            if i < len(hidden_layers) - 1:  # 只在非最后一层之前添加Dropout
                deep_layers.append(nn.Dropout(0.5))
        
        # 使用Sequential构建Deep部分
        self.deep = nn.Sequential(*deep_layers)

    def forward(self, X_wide, X_deep):
        wide_out = self.wide(X_wide)
        deep_out = self.deep(X_deep)
        combined = torch.cat([wide_out, deep_out], 1)
        return torch.sigmoid(combined)
