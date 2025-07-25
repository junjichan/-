import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from model import Relevance_bert
from dataset import TrainDataset
from args import parse_args
import numpy as np
from sklearn.metrics import f1_score

# 单张图片分析模式
def predict_single(model, tokenizer, sentence1, sentence2, device):
    """
    单个样本推理
    :param model: 训练好的模型
    :param tokenizer: 文本tokenizer
    :param sentence1: 第一个句子
    :param sentence2: 第二个句子
    :param device: 计算设备
    :return: 预测结果
    """
    model.eval()
    
    text = str(sentence1) + "[SEP]" + str(sentence2)
    text_enc = tokenizer(text, truncation=True, padding='max_length', 
                        max_length=130, return_tensors="pt")
    
    with torch.no_grad():
        input_ids = text_enc['input_ids'].to(device)
        attention_mask = text_enc['attention_mask'].to(device)
        
        outputs, prob, _ = model(input_ids, attention_mask)
        _, predicted = torch.max(prob, 1)
        
    label_mapping = {0: 'not_match', 1: 'partial_match', 2: 'exact_match'}
    return label_mapping[predicted.item()], prob.squeeze().tolist()


def evaluate(model, eval_loader, device):
    """
    评估模型性能
    :param model: 训练好的模型
    :param eval_loader: 评估数据加载器
    :param device: 计算设备
    :return: 评估结果
    """
    model.eval()
    eval_loss = 0.00
    eval_accuracy = 0.00
    eval_total_num = 0
    eval_correct_preds = 0
    eval_all_preds = []
    eval_all_labels = []
    
    with torch.no_grad():
        for index, (input_ids, attention_mask, labels) in tqdm(enumerate(eval_loader), 
                                                              total=len(eval_loader), 
                                                              desc="Evaluation Progress"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs, prob, _ = model(input_ids, attention_mask)
            
            # 计算损失
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            
            # 计算预测正确的个数
            _, predicted = torch.max(prob, 1)
            eval_correct_preds += (predicted == labels).sum().item()
            eval_total_num += labels.size(0)
            
            # 收集预测和标签
            eval_all_preds.extend(predicted.cpu().numpy())
            eval_all_labels.extend(labels.cpu().numpy())
    
    eval_loss = eval_loss / len(eval_loader)
    eval_accuracy = (eval_correct_preds / eval_total_num) * 100
    eval_f1 = f1_score(eval_all_labels, eval_all_preds, average='macro') * 100
    
    result = {
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,
        'eval_f1': eval_f1
    }
    
    return result

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 解析参数
    args = parse_args()
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)
    
    # 加载模型
    model = Relevance_bert(num_classes=3).to(device)
    model.load_state_dict(torch.load(args.model_path))
    
    # 根据参数选择模式
    if args.mode == "single":
        # 单个样本推理模式
        sentence1 = "句子1"
        sentence2 = "句子2"
        
        pred_label, pred_probs = predict_single(model, tokenizer, sentence1, sentence2, device)
        print(f"预测结果: {pred_label}")
        print(f"各类别概率: not_match={pred_probs[0]:.4f}, partial_match={pred_probs[1]:.4f}, exact_match={pred_probs[2]:.4f}")
    else:
        # 批量评估模式
        eval_dataset = TrainDataset(args.eval_path, file_mode="csv")
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
        
        eval_result = evaluate(model, eval_loader, device)
        
        print(f"Evaluation Results:")
        print(f"Loss: {eval_result['eval_loss']:.4f}")
        print(f"Accuracy: {eval_result['eval_accuracy']:.2f}%")
        print(f"F1 Score: {eval_result['eval_f1']:.2f}%")
