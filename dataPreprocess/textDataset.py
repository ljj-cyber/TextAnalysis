import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
from .data import *


class textDataset(Dataset):
    def __init__(self, config, rawData: str) -> None:
        """
        加载config配置文件
        如果采用预训练模式，直接调库生成input_id
        如果采用自己的embedding，使用字典生成input_ids
        Args:
            config (_type_): 配置文件
            rawData (str): 数据集路径
        """        
        self.config = config
        self.rawData = pd.read_csv(rawData)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese') 


    def __len__(self):
        return len(self.rawData)
    

    def __getitem__(self, index):
        """
        根据训练模式，如果是训练模式，返回(word_ids, label)
        如果是测试模式，返回word_ids
        Args:
            index (_type_): _description_
        Returns:
            Any: _description_
        """        
        text_tokens = dataCleaning(self.rawData['text'][index])
        # 根据训练模式进行处理
        if self.config.training:
            # 如果是训练模式，返回(word_ids, label)
            label = self.rawData['label'][index]  
            return self.process_text(text_tokens), label
        else:
            # 如果是测试模式，只返回word_ids
            return self.process_text(text_tokens)
        

    def process_text(self, text_tokens):
        if self.config.use_pretrained_embedding:
            # 如果使用预训练模型，直接调用Transformers库生成input_ids
            inputs = self.tokenizer(text_tokens, return_tensors='pt', padding=True, truncation=True)
            return inputs['input_ids'].squeeze()
        else:
            # 如果使用自己的embedding，使用字典生成input_ids
            word_ids = [self.config.word2id.get(word, self.config.unk_token_id) for word in text_tokens]
            return torch.tensor(word_ids)    
        

