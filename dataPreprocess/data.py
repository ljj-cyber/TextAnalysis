import pandas as pd
import re
import jieba
from jieba import posseg


def dataCleaning(filename: str):
    """
    对每条数据进行数据清洗，并分词
    Args:
        filename (str): 文件路径名
    Returns:
        list[list[str]]: 分词结果
    """    
    """
    
    :param filename: the absolute/relative path of the data file
    :return: shape [#comments, #words]
    """
    # 读取CSV文件
    data = pd.read_csv(filename, names=['text', 'label'])

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

    words = []
    for index, row in data.iterrows():
        # 对第一列文字数据进行去除停用词和分词
        words.append(remove_stopwords_and_segment(row[0]))

    return words


def splitDataset(allDataPath: str):
    """
    将整个数据集划分成训练集、测试集、验证集，并存下来
    Args:
        allDataPath (str): 总数据集路径
    """    


if __name__ == '__main__':
    print(dataCleaning('./data/comments.csv'))