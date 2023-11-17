from dataPreprocess.textDataset import *
from model.config import Config
import pandas as pd

if __name__ == '__main__':
    # # 使用示例
    # config = Config()
    # dataset = textDataset(config, rawData='test_dataset.csv')
    # print(dataset)
    # sample = dataset[0]

    # # 如果是训练模式，返回(word_ids, label)
    # if config.training:
    #     word_ids, label = sample
    #     print(word_ids.shape)
    #     print(f"Word IDs: {word_ids}, Label: {label}")
    # else:
    #     # 如果是测试模式，只返回word_ids
    #     word_ids = sample
    #     print(f"Word IDs: {word_ids}")

    train_data = pd.read_csv('train_dataset.csv')
    print(pd.value_counts(train_data['label']))
    test_data = pd.read_csv('test_dataset.csv')
    print(pd.value_counts(test_data['label']))
    val_data = pd.read_csv('val_dataset.csv')
    print(pd.value_counts(val_data['label']))