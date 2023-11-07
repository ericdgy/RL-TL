import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 数据预处理
df = pd.read_csv('')
features = df.drop(['Label'], axis=1).values  # 假设最后一列是标签
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
dataset = TensorDataset(torch.tensor(scaled_features).float())
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


# 定义Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_size),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, input_size),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


input_size = scaled_features.shape[1]
encoding_size = 32  # 可以根据需要调整

autoencoder = Autoencoder(input_size, encoding_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

# 训练Autoencoder
epochs = 50
for epoch in range(epochs):
    for data in dataloader:
        inputs = data[0]
        # 正向传播
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 特征提取
encoded_samples = []
for data in dataloader:
    inputs = data[0]
    encoded = autoencoder.encoder(inputs)
    encoded_samples.append(encoded.detach().numpy())

# 连接所有的编码后的数据
encoded_features = np.concatenate(encoded_samples, axis=0)
