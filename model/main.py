import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
import sys
sys.path.append(r"c:\Users\11276\Desktop\textAnalysis")
import os
from sklearn import metrics
from dataPreprocess import textDataset
from model import Config, Model
import numpy as np
import time


def train(config):
    # 超参数
    batch_size = config.batch_size
    learning_rate = config.bert_learning_rate
    epochs = config.num_epochs

    # 加载数据集
    train_set = textDataset(config, './train_dataset.csv')
    # 若shuffle=True,则sampler=RandomSampler(dataset)
    train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True)

    val_set = textDataset(config, './val_dataset.csv')
    val_loader = DataLoader(val_set, batch_size, shuffle=True, drop_last=True)

    # 初始化模型、损失函数和优化器
    model = Model(config)  # 你的模型类
    model.to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5, last_epoch=-1)  # 每2个epoch学习率衰减为原来的一半
    # 训练循环
    dev_best_loss = 10
    for epoch in range(epochs):
        if os.path.exists(config.save_path):
            model.load_state_dict(torch.load(config.save_path))

        model.train()
        total_loss = 0

        for i, data in enumerate(train_loader):
            inputs, labels= data
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失, 值得注意的点，这里的label不需要赋值one-hot编码类型，因为函数内部会自动将label变换为one-hot类型
            loss = F.cross_entropy(outputs, labels, weight=torch.tensor([10.0,5.0,3.0,2.0,1.0]).to(config.device))

            pred = torch.max(outputs, dim=1)[1]
            pred = pred.cpu()
            # torch.eq(pred, labels)
            true_label = labels.data.cpu()
            train_acc = metrics.accuracy_score(true_label, pred)

            # 反向传播和优化
            loss.backward()
            optimizer.step()
            # lr = optimizer.state_dict()['param_groups'][0]['lr']
            # print(f'learning rate: {lr}')
            scheduler.step()

            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}, Train accuracy: {train_acc:>6.4%}")    

        if epoch % 5 == 0:
            dev_acc, dev_loss = evaluate(model, val_loader, test=False)  # model.eval()
            if dev_loss < dev_best_loss:
                dev_best_loss = dev_loss
                torch.save(model.state_dict(), config.save_path)
                improve = '*'
                print(f"Epoch {epoch + 1}/{epochs}, dev Loss: {dev_loss}, dev accuracy: {dev_acc:>6.4%}")    

            else:
                improve = ''

    


def test(config, model):
    test_set = textDataset(config, './test_dataset.csv')
    test_loader = DataLoader(test_set, config.batch_size, shuffle=True, drop_last=True)
    # 测试函数
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    # start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_loader, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...") #精确率和召回率以及调和平均数
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    # time_dif = get_time_dif(start_time)
    # print("Time usage:", time_dif)


def evaluate(model, data_iter, test):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, data in enumerate(data_iter):
            inputs, labels = data
            # print(texts)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, weight=torch.tensor([10.0,5.0,3.0,2.0,1.0]).to(config.device))
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()  ###预测结果
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


if __name__ == '__main__':
    config = Config()
    if config.training == True:
        train(config)
    else:
        model = Model(config)  # 你的模型类
        model.to(config.device)
        test(config, model)
