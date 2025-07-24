import pandas as pd
import numpy as np


import json
import re





if __name__ == "__main__":

    df = pd.read_csv("./dataset/train/train.csv")
    # 数据清洗 缺失值删除
    count = 0
    new_df = []
    for index,row in df.iterrows():
        if type(row["sentence1"]) != str or type(row["sentence2"]) != str:
            count = count + 1
        else:
            new_df.append(row)
    print("删除缺失值数量为:",count)
    new_df = pd.DataFrame(new_df)
    new_df.to_csv("./dataset/train/train.csv",index=False)


            

