import torch
import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = BertModel.from_pretrained("./minirbt-h256")
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.embedding(x) 
        out, (hidden_state, cell_memory) = self.lstm(out[0])
        hidden_state = torch.mean(hidden_state, dim=0)
        out = self.fc(hidden_state)  # 句子最后时刻的 hidden state
        out_probs = self.softmax(out)  # Apply softmax to get probabilities
        return out
