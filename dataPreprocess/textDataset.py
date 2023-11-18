import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd

import sys
sys.path.append(r"c:\Users\11276\Desktop\textAnalysis")
from data import dataCleaning


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
        self.rawData = pd.read_csv(rawData, engine='python')
        self.dataCleaning()
        self.tokenizer = BertTokenizer.from_pretrained("./minirbt-h256") 


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
        try:
            text = self.rawData['text'][index]
            label = self.rawData['label'][index]  
            label = label - 1
        except ValueError as e:
            print(text, label)
        return self.process_text(text), torch.tensor(label, dtype=torch.long).to(self.config.device)
        

    def dataCleaning(self):
    # 假设self.rawData是一个DataFrame，且包含两列需要处理的数据
        if 'text' in self.rawData and 'label' in self.rawData:
            # 删除包含缺失值的行
            self.rawData = self.rawData.dropna(subset=['text', 'label']).reset_index(drop=True)
            # 随机采样比例为0.1的数据
            self.rawData = self.rawData.sample(frac=self.config.frac, random_state=1).reset_index(drop=True)

    def process_text(self, text):
        # attention_mask = torch.tensor(text['attention_mask'], dtype=torch.long)
        if self.config.use_pretrained_embedding:
            try:
                # 使用 tokenizer.encode_plus 处理截断和填充
                inputs = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.config.pad_size,
                    padding='max_length',
                    truncation=True,
                )
                return torch.tensor(inputs['input_ids'], dtype=torch.long).to(self.config.device)
            except ValueError as e:
                print(text)
        # else:
        #     # 如果使用自己的embedding，使用字典生成input_ids
        #     word_ids = [self.config.word2id.get(word, self.config.unk_token_id) for word in text_tokens]
        #     return torch.tensor(word_ids)    