import torch.nn as nn
from transformers import BertModel

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = BertModel.from_pretrained('bert-base-chinese')
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 128, 768]
        out, _ = self.lstm(out[1])
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        out_probs = self.softmax(out)  # Apply softmax to get probabilities
        return out_probs
