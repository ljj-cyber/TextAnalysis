import torch
import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = BertModel.from_pretrained("./minirbt-h256")
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            batch_first=True, dropout=config.dropout)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.embedding(x)
        
        # 提取[CLS]标记的隐藏状态
        cls_token_embedding = out['last_hidden_state'][:, 0, :]
        
        # LSTM层
        lstm_out, _ = self.lstm(cls_token_embedding.unsqueeze(1))
        
        # 使用attention机制聚合所有隐藏状态
        attention_weights = torch.softmax(torch.matmul(lstm_out.squeeze(), lstm_out.squeeze().transpose(-1, -2)), dim=-1)
        context_vector = torch.matmul(attention_weights, lstm_out.squeeze())
        
        # 全连接层处理聚合的隐藏状态
        out = self.fc1(context_vector)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        out_probs = self.softmax(out)
        return out_probs
