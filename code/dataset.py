import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset  # PyTorch数据集基类
from torchvision import transforms   # 图像预处理工具
from PIL import Image, ImageEnhance, ImageFilter  # 图像处理库
from transformers import BertTokenizer, BertModel

import random
import re
from args import parse_args

args = parse_args()

def replace_with_synonyms(text):
        """
        基于词表进行同义词替换
        :param text: 输入文本
        :return: 替换后的文本
        """
        # 示例同义词词表，可根据需要扩展
        synonym_dict = {
            # 道路类型
            "路": ["街", "道", "大街", "大道"],
            "街": ["路", "道", "巷"],
            "大道": ["路", "大街", "干线"],
            "巷": ["弄", "胡同", "里"],
        
            # 建筑单元
            "幢": ["栋", "号楼", "座"],
            "栋": ["幢", "号楼", "座"],
            "号楼": ["幢", "栋", "座"],
            "座": ["幢", "栋", "号楼"],

            "单元": ["梯", "区"],
        
            # 区域类型
            "小区": ["花园", "苑", "新村", "社区"],
            "花园": ["小区", "苑", "园"],
            "苑": ["小区", "花园", "新村"],
            "社区": ["小区", "居委会", "街道"],
        
            # 行政单位
            "街道": ["镇", "办事处", "街办"],
            "镇": ["街道", "乡"],
        
            # 门牌标识
            "楼": ["号楼", "大厦"],
            "门": ["出口", "出入口"],
        
            "开发区": ["产业园", "工业区"],

            # 数字替换策略
            "0": ["零"],
            "1": ["一"],
            "2": ["二"],
            "3": ["三"],
            "4": ["四"],
            "5": ["五"],
            "6": ["六"],
            "7": ["七"],
            "8": ["八"],
            "9": ["九"],

            "零": ["0"],
            "一": ["1"],
            "二": ["2"],
            "三": ["3"],
            "四": ["4"],
            "五": ["5"],
            "六": ["6"],
            "七": ["7"],
            "八": ["8"],
            "九": ["9"]
        }
        
        words = list(text)
        for i in range(len(words)):
            if words[i] in synonym_dict and random.random() < 0.5:  # 30%概率替换
                synonyms = synonym_dict[words[i]]
                words[i] = random.choice(synonyms)
        
        return ''.join(words)


class TrainDataset(Dataset):
    def __init__(self, data_dir, transform=None, file_mode = args.file_mode,augment = None):
        self.transform = transform
        self.augment = augment

        if file_mode == "csv":
            self.data = pd.read_csv(data_dir)
        else:
            raise ValueError("Invalid file mode: {}".format(file_mode))
        
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)

        self.label_mapping = {'not_match':0, 'partial_match':1,'exact_match':2}
        self.label_list = []
        self.sentence1_list = []
        self.sentence2_list = []
        for i in range(self.data.shape[0]):
            item = self.data.iloc[i]
            self.label_list.append(self.label_mapping[item['label']])
            self.sentence1_list.append(item['sentence1'])
            self.sentence2_list.append(item['sentence2'])
        
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.label_list)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        :param idx: 样本索引
        :return: (图像张量, 文件名)
        """
        sentence1 = self.sentence1_list[idx]
        sentence2 = self.sentence2_list[idx]
        label = self.label_list[idx]
        if self.augment:
            # 数据增强1：句子顺序交换
            if random.random() < 0.5:
                sentence1, sentence2 = sentence2, sentence1
                
            # # 数据增强2,同义词替换增强
            # if random.random() < 0.3:
            #     sentence1 = replace_with_synonyms(sentence1)
            # if random.random() < 0.3:
            #     sentence2 = replace_with_synonyms(sentence2)

            # # 数据增强3，括号内容删除
            # if random.random() < 0.3:
            #     if contains_brackets(sentence1):
            #         sentence1 = re.sub(r'\(.*?\)', '', sentence1)
            #     if contains_brackets(sentence2):
            #         sentence2 = re.sub(r'\(.*?\)', '', sentence2)
        #print(sentence1)
        #print(sentence2)
        text = str(sentence1) + "[SEP]" + str(sentence2)
        
        text_enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=130, return_tensors="pt")
        return text_enc['input_ids'].squeeze(0), text_enc['attention_mask'].squeeze(0), label  
        # 返回图像和文件名（无标签）


def contains_brackets(text):
    # 正则表达式，匹配圆括号、方括号或花括号
    pattern = r'[\(\)\[\]\{\}]'
    
    # 使用 re.search 来查找文本中是否有括号
    if re.search(pattern, text):
        return True
    else:
        return False
    

if __name__ == "__main__":
    # 示例用法
    train_dir = "/hy-tmp/dataset/train/train.csv" 
    train_dataset = TrainDataset(train_dir)
    print(train_dataset.__len__())
