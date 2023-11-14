from dataPreprocess.textDataset import *
from model.config import Config

if __name__ == '__main__':
    # 使用示例
    config = Config()
    dataset = textDataset(config, rawData='test_dataset.csv')
    print(dataset)
    sample = dataset[0]

    # 如果是训练模式，返回(word_ids, label)
    if config.training:
        word_ids, label = sample
        print(word_ids.shape)
        print(f"Word IDs: {word_ids}, Label: {label}")
    else:
        # 如果是测试模式，只返回word_ids
        word_ids = sample
        print(f"Word IDs: {word_ids}")