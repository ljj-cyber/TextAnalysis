import pandas as pd
import re
import jieba
from jieba import posseg
from sklearn.model_selection import train_test_split


def dataCleaning(text: str):
    """
    对每条数据进行数据清洗，并分词
    Args:
        text (str): 文本数据
    Returns:
        list[list[str]]: 分词结果
    """    
    """
    
    :param filename: the absolute/relative path of the data file
    :return: shape [#comments, #words]
    """
    # 读取停用词列表
    stopwords = []  # 根据实际情况添加停用词
    
    with open('hit_stopwords.txt', encoding='utf-8') as f:
        stopwords = [line.replace("\n", "") for line in f.readlines()]
    
    # 定义正则表达式模式，用于过滤无用字符
    pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9]+'  # 只保留中文字符、英文字符和数字

    # 清洗数据
    def clean_text(text):
        # 去除多余的空格
        text = text.strip()
        # 过滤无用字符
        text = re.sub(pattern, '', text)
        return text

    # 去除停用词和分词
    def remove_stopwords_and_segment(text):
        # 清洗数据
        cleaned_text = clean_text(text)
        # 分词
        seg_list = jieba.cut(cleaned_text)
        # 去除停用词
        seg_list_without_stopwords = [word for word in seg_list if word not in stopwords]
        # 返回分词结果
        return seg_list_without_stopwords

    # 文字数据进行去除停用词和分词
    return remove_stopwords_and_segment(text)


def splitDataset(allDataPath: str, train_size: float = 0.8, test_size: float = 0.1, random_state: int = 42):
    """
    将整个数据集划分成训练集、测试集、验证集，并存储下来
    Args:
        allDataPath (str): 总数据集路径
        train_size (float): 训练集比例，默认为0.8
        test_size (float): 测试集比例，默认为0.1
        random_state (int): 随机种子，默认为42
    """

    # 读取CSV文件
    df = pd.read_csv(allDataPath, header=None, names=['text', 'label'])

    # 获取标签和特征
    labels = df['label']
    features = df['text']

    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, train_size=train_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, random_state=random_state)

    # 存储划分后的数据集
    saveDataset(X_train, y_train, "train_dataset.csv")
    saveDataset(X_val, y_val, "val_dataset.csv")
    saveDataset(X_test, y_test, "test_dataset.csv")
    

def saveDataset(X, y, outputPath):
    """
    存储数据集到CSV文件
    Args:
        X: 特征列表
        y: 标签列表
        outputPath: 输出文件路径
    """
    df = pd.DataFrame({'text': X, 'label': y})
    df.to_csv(outputPath, index=False)




if __name__ == '__main__':
    # print(dataCleaning('./data/test.csv'))
    splitDataset("./dataset/test.csv")