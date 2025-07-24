import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

from args import parse_args

args = parse_args()

class Relevance_bert(nn.Module):
    def __init__(self, num_classes=args.num_classes):
        super().__init__()
        self.text_backbone = BertModel.from_pretrained(args.pretrained_path)
        self.text_proj = nn.Linear(768, 256)
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, x, attention_mask):

        x = self.text_backbone( input_ids= x, attention_mask=attention_mask).pooler_output
        projected_emb  = self.text_proj(x)
        x = self.classifier(projected_emb )
        logits = self.classifier(projected_emb)
        prob = F.softmax(logits, dim=1)
        return  x,prob, F.normalize(projected_emb, p=2, dim=1)

if __name__ == '__main__':
    model = Relevance_bert()
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)
    text1 = "地址1"
    text2 = "地址2"

    text = text1 + "[SEP]" + text2  
    print("文本的长度为：",len(text1))
    text_enc = tokenizer(text, truncation=True, padding='max_length', max_length=130, return_tensors="pt")
    print(type(text_enc))
    print(text_enc.keys())
    print(text_enc['input_ids'])
    print(text_enc['input_ids'].shape)



    # print(text_enc['attention_mask'])
    # print(text_enc['attention_mask'].shape)
    # print(text_enc['token_type_ids'])
    # print(text_enc['token_type_ids'].shape)
#        text_enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=32, return_tensors="pt")

#       label = row['label']  # 已映射为整数标签
#       return image, text_enc['input_ids'].squeeze(0), text_enc['attention_mask'].squeeze(0), label


    # mask = torch.ones((1,10))
    # y = model(x,mask)
    # print(y.shape)