import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_class_alpha(label_list, num_classes):
    # 统计每个类别的样本数量
    # 初始化一个长度为num_classes的列表
    class_counts = [0] * num_classes  
    for label in label_list:
            class_counts[label] += 1
    
    # 计算样本总数
    total_samples = sum(class_counts)
    
    # 计算每个类别的权重（alpha），避免除以零
    alpha = []
    for count in class_counts:
        if count > 0:
            alpha.append(total_samples / count)  # 正常计算权重
        else:
            alpha.append(0)  # 对于没有样本的类别，设置为0或者其它默认值
    
    # 将 alpha 转换为 tensor 格式
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
    
    return alpha_tensor

class CEFL(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super(CEFL, self).__init__()
        self.alpha = alpha  # 类别的权重
        self.gamma = torch.tensor(gamma, dtype=torch.float32).to(device) 
        self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device) 
        print("CEFL中的gamma变量在设备",device,"上")
    def forward(self, inputs, targets):
        inputs = inputs.to(device)
        targets = targets.to(device)
                
        
        # 使用softmax计算类别概率
        p = F.softmax(inputs, dim=1)
        
        # 选择正确类别的预测概率
        p_t = p.gather(1, targets.view(-1, 1))
    
        # 计算损失
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t)
        
        return loss.mean()

def compute_class_frequencies(targets, num_classes):
    # 计算每个类别的样本数量
    class_counts = np.bincount(targets, minlength=num_classes)
    
    # 防止除零错误，计算每个类别的频率
    class_freq = 1.0 / (class_counts + 1e-6)
    
    # 归一化类别频率
    class_freq = class_freq / np.sum(class_freq)
    
    return torch.tensor(class_freq, dtype=torch.float32).to(device) 

class CEFL2(nn.Module):
    def __init__(self, class_frequencies, gamma=2.0):
        super(CEFL2, self).__init__()
        self.class_frequencies = class_frequencies  # 类别频率
        self.gamma = gamma  # 焦点损失的调节参数
        self.gamma = torch.tensor(gamma, dtype=torch.float32).to(device) 
    def forward(self, inputs, targets):
        inputs = inputs.to(device)
        targets = targets.to(device)
        # 使用softmax计算类别概率
        p = F.softmax(inputs, dim=1)
        
        # 选择正确类别的预测概率
        p_t = p.gather(1, targets.view(-1, 1))

        # 计算每个类别的加权损失
        loss_term_1 = (1 - p_t)**2 / ((1 - p_t)**2 + p_t**2) * torch.log(p_t)
        loss_term_2 = p_t**2 / ((1 - p_t)**2 + p_t**2) * (1 - p_t)**self.gamma * torch.log(p_t)
        
        # 将每个类别的频率作为加权项
        loss = -self.class_frequencies[targets] * (loss_term_1 + loss_term_2)
        
        return loss.mean()

def build_positive_pairs(embeddings, labels):
    """
    根据相同标签动态构建正样本对
    :param embeddings: 当前batch的嵌入向量 [batch_size, dim]
    :param labels: 当前batch的标签 [batch_size]
    :return: (anchor, positive) 正样本对
    """
    labels = labels.cpu().numpy().astype(int)
    unique_labels = np.unique(labels)
    
    anchors, positives = [], []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) < 2:
            continue
            
        # 为每个样本随机配对另一个同标签样本
        np.random.shuffle(indices)
        for i in range(len(indices)):
            j = (i + 1) % len(indices)
            anchors.append(embeddings[indices[i]])
            positives.append(embeddings[indices[j]])
    
    return torch.stack(anchors), torch.stack(positives) if anchors else (None, None)
def nce_loss(anchor, positive, temperature=0.1):
    """
    anchor: 当前样本向量 [batch_size, dim]
    positive: 正样本向量 [batch_size, dim]
    temperature: 温度系数
    """
    # 计算正样本相似度
    pos_sim = torch.sum(anchor * positive, dim=-1) / temperature  # [batch_size]
    
    # 计算负样本相似度（同一batch内其他样本作为负例）
    neg_sim = torch.mm(anchor, positive.T) / temperature  # [batch_size, batch_size]
    
    # 对角线是正样本，需要排除
    neg_sim = neg_sim.masked_fill(torch.eye(anchor.size(0)).bool().to(anchor.device), -1e9)
    
    # 合并计算NCE损失
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=1)  # [batch, batch_size]
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(anchor.device)
    return F.cross_entropy(logits, labels)

    