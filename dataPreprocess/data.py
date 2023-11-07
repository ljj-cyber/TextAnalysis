import pandas as pd
import re
import jieba
from jieba import posseg


class dataPreprocess:
    def dataCleaning(self):
        # 读取CSV文件
        data = pd.read_csv('../data/comments.csv')

        # 定义停用词列表
        stopwords = ['的', '了', '是', '我', '你', '他', '她']  # 根据实际情况添加停用词

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

        # 对第一列文字数据进行去除停用词和分词
        data['文字'] = data['文字'].apply(remove_stopwords_and_segment)

        # 打印处理后的数据
        print(data)


if __name__ == '__main__':
    data = pd.read_csv('../data/comments.csv')
    print(data.shape)
