import os.path
import torch


class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5     # 随机失活
        self.require_improvement = 1500  # 若超过2000batch效果还没提升，则提前结束训练
        self.num_classes = 2  # 类别数无需修改
        self.num_epochs = 50   # epoch数
        self.batch_size = 64  # mini-batch大小，看显存决定
        self.pad_size = 128  # 每句话处理成的长度(短填长切)
        self.bert_learning_rate = 1e-4   # bert的学习率，minirbt-h256需要用更大的学习率例如1e-4,其他bert模型设置为1e-5较好
        self.tokenizer = ''
        self.frac = 1  # 使用数据的比例，因为训练时间长，方便调参使用,1为全部数据，0.1代表十分之一的数据
        #['bert_model/bert-base-chinese','bert_model/chinese-bert-wwm-ext','bert_model/minirbt-h256']
        self.bert_name = '../bert_model/minirbt-h256'  # bert类型，三种选择
        self.bert_fc = 256  # bert全连接的输出维度 bert-base-chinese和chinese-bert-wwm-ext为768，minirbt-h256为256

        if not os.path.exists('../model_stored'):
            os.makedirs('../model_stored')

        self.use_pretrained_embedding = True
        self.training = True