import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
import sys
sys.path.append(r"c:\Users\11276\Desktop\textAnalysis")

from dataPreprocess import textDataset
from model import Config, Model

def train(config):
    # 超参数
    batch_size = config.batch_size
    learning_rate = config.bert_learning_rate
    epochs = config.num_epochs

    # 数据路径
    raw_data_path = "./test_dataset.csv"

    # 加载数据集
    dataset = textDataset(config, raw_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = Model(config)  # 你的模型类
    model.to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, data in enumerate(dataloader):
            inputs, labels= data
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = F.cross_entropy(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}")

    # 保存训练好的模型
    torch.save(model.state_dict(), "trained_model.pth")


if __name__ == '__main__':
    train(Config())