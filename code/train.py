import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
# 导入自定义的类和模块
from model import Relevance_bert 
from dataset import TrainDataset  # 假设你的模型和数据集类在 model.py 中
from loss import compute_class_frequencies,compute_class_alpha,CEFL,CEFL2,build_positive_pairs,nce_loss
import pandas as pd
from args import parse_args
import numpy as np
from sklearn.metrics import f1_score 
# 设置随机种子确保结果可复现
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)
args = parse_args()
def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):
    
    model.train()  # 将模型设置为训练模式
    epoch_list = []
    train_loss_list = []
    train_accuracy_list = []
    train_f1_list = []
    val_loss_list = []
    val_accuracy_list = []
    val_f1_list = [] 
    best_val_loss = 1000
    for epoch in range(epochs):
        epoch_list.append(epoch)
        train_loss = 0.00  
        train_accuracy = 0.00      
        train_total_num = 0
        train_correct_preds = 0
        train_all_preds = []   # 存储所有训练预测
        train_all_labels = []  # 存储所有训练标签
        for index, (input_ids, attention_mask, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Progress"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs, prob, embeddings = model(input_ids, attention_mask)
            # 计算交叉熵损失
            ce_loss = criterion(outputs, labels)
            
            if args.NCE_function == True:
                # 动态构建正样本对
                anchor, positive = build_positive_pairs(embeddings, labels)
                
                if anchor is not None:  # 如果成功构建正样本对
                    nce_loss_value = nce_loss(anchor, positive, temperature=0.07)
                    loss = 0.7 * ce_loss + 0.3 * nce_loss_value  # 混合损失
                else:
                    loss = ce_loss  # 回退到纯交叉熵
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #计算训练损失
            train_loss += ce_loss.item()

            # 计算预测正确的个数
            _, predicted = torch.max(outputs, 1)
            train_correct_preds += (predicted == labels).sum().item()
            train_total_num += labels.size(0)

            
            # 收集当前batch的预测和标签（用于计算F1）
            train_all_preds.extend(predicted.cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())
        
        train_loss = train_loss / len(train_loader)
        train_accuracy = (train_correct_preds / train_total_num) * 100
        train_f1 = f1_score(train_all_labels, train_all_preds, average='macro') * 100
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        train_f1_list.append(train_f1)  # 保存F1分数

        model.eval()
        val_loss = 0.00  
        val_accuracy = 0.00
        val_total_num = 0
        val_correct_preds = 0
        val_all_preds = []   # 存储所有验证预测
        val_all_labels = []  # 存储所有验证标签
        with torch.no_grad():
            for index, (input_ids, attention_mask, labels) in tqdm(enumerate(val_loader), total=len(val_loader), desc="val Progress"):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outputs,prob,embedding = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            
                val_loss += loss.item()

                # 计算预测正确的个数
                _, predicted = torch.max(prob, 1)
                val_correct_preds += (predicted == labels).sum().item()
                val_total_num += labels.size(0)

                # 收集当前batch的预测和标签（用于计算F1）
                val_all_preds.extend(predicted.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())
                
                
       
        val_loss = val_loss / len(val_loader)
        val_accuracy = (val_correct_preds / val_total_num) * 100
        
        # 计算验证集的Micro F1分数
        val_f1 = f1_score(val_all_labels, val_all_preds, average='macro') * 100
        
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)
        val_f1_list.append(val_f1)  # 保存F1分数
        
        print(f"Epoch [{epoch + 1}], "f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Train F1: {train_f1:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val F1: {val_f1:.2f}%")
        
        # 如果验证损失改善，保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path+f'best_model_{epoch+1}_{val_loss:.4f}.pth')
    
    train_process = pd.DataFrame({
        'epoch': epoch_list,
        'train_loss': train_loss_list,
        'train_accuracy': train_accuracy_list,
        'train_f1': train_f1_list,      # 添加F1分数
        'val_loss': val_loss_list,
        'val_accuracy': val_accuracy_list,
        'val_f1': val_f1_list           # 添加F1分数
    })
    train_process.to_csv("./train_process.csv",index=False)
    
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据集
    train_dir = args.train_path  # 假设数据文件在这个路径
    val_dir = args.val_path
    train_dataset = TrainDataset(train_dir,file_mode = "csv",augment = args.augment)
    val_dataset = TrainDataset(val_dir,file_mode = "csv")

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 初始化模型
    model = Relevance_bert(num_classes=3).to(device)
    
    # 定义损失函数
    if args.loss_function == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
        
    elif args.loss_function == "CEFL":
        label_list = train_dataset.label_list
        alpha = compute_class_alpha(label_list,args.num_classes)
        criterion = CEFL(alpha)

    elif args.loss_function == "CEFL2":
        targets = np.array(train_dataset.label_list)
        class_frequencies = compute_class_frequencies(targets, args.num_classes)
        criterion = CEFL2(class_frequencies)
        
    
    # 定义优化器
    #optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-3)
    # 训练模型
    train(model, train_loader, val_loader, criterion, optimizer, device, epochs=args.epochs)
