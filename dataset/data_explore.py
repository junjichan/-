import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
plt.rcParams['font.sans-serif'] = ['SimHei']
def data_explore(mode="train"):
    feature_set=set()
    df = pd.DataFrame(columns=["sentence1","sentence2","label"])
    sentence1 = []
    sentence2 = []
    label = []
    with open(f'./dataset/{mode}/{mode}.json','r',encoding='utf-8') as file:
        content = file.read()
        objects = content.split('\n')
        for obj in objects:
            if obj.strip():  
                data = json.loads(obj)
                feature_set.update(data.keys())
                sentence1.append(data['sentence1'])
                sentence2.append(data['sentence2'])
                label.append(data['label'])
    df["sentence1"] = sentence1
    df["sentence2"] = sentence2
    df["label"] = label
    print("数据集大小为：",len(df))
    print("数据集中的键值有：")
    print(feature_set)
    print("sentence1的取数数量为:",df["sentence1"].nunique())
    print("sentence2的取数数量为:",df["sentence2"].nunique())
    print("label的取数数量为:",df["label"].nunique())
    print("label的取数有:",df["label"].unique())
    df.to_csv(f"./dataset/{mode}/{mode}.csv",index=False,encoding='utf-8')
    return df
def show_length(df,num=None):
    max_length = 0
    length = []
    long_text = []
    for index,row in df.iterrows():
        length.append(len(row["sentence1"])+len(row["sentence2"]))
        if len(row["sentence1"])+len(row["sentence2"])>max_length:
            max_length = len(row["sentence1"])+len(row["sentence2"])
            max_text1 = row["sentence1"]
            max_text2 = row["sentence2"]
        if num != None and length[-1]>=num:
            long_text.append(row["sentence1"]+","+row["sentence2"])
    if num != None:
        long_text = pd.DataFrame(long_text,columns=["text"]) 
        long_text.to_csv("long_text.csv",index=False)



            
    return max_length,length,max_text1,max_text2
def draw_length(length,description=None):
    if description is None:
        title = "句子长度分布"
    else:
        title = description + "句子长度分布"
    plt.hist(length, bins=130, edgecolor='black')
    plt.title(title)
    plt.xlabel("句子长度")
    plt.ylabel("频数")
    plt.show()
    
def draw_label(df,text=None):
    """
    可视化展示句子类别的分布情况
    :param df: 包含label列的DataFrame
    """
    # 统计各类别数量
    label_counts = df['label'].value_counts()
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制饼图
    ax1.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', 
           startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    if text != None:
        ax1.set_title(text+'标签分布(饼图)')
    else:
        ax1.set_title('标签分布(饼图)')
    ax1.axis('equal')  # 保证饼图是圆形
    
    # 绘制柱状图
    sns.barplot(x=label_counts.index, y=label_counts.values,ax=ax2,palette=['#ff9999','#66b3ff','#99ff99'])
    if text != None:
        ax2.set_title(text+'标签分布(柱状图)')
    else:
        ax2.set_title('标签分布(柱状图)')
    
    ax2.set_ylabel('数量')
    # 调整柱状图的标签角度
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

def sentence_letter(sentence):
    lower_count = sum(1 for char in sentence if char.islower())
    upper_count = sum(1 for char in sentence if char.isupper())
    return lower_count+upper_count

    
def data_clean(df):
    cleaned_data = []
    clean_data = []
    letter_count_list = []
    # 遍历 DataFrame 中的每一行
    for _, row in df.iterrows():
        # 1.我们发现sentence中含有与地址信息大量无关的备注，通常采用了括号的方式标志
        # 为了避免引入额外的噪音，我们对sentence中含有括号部分进行了消除
        sentence1 = row['sentence1']
        sentence2 = row['sentence2']
        row['sentence1'] = re.sub(r'\(.*?\)', '', sentence1)
        row['sentence1'] = re.sub(r'\(.*?\)', '', sentence2)

        # 1.我们发现sentence中含有与地址信息大量无关的备注，通常采用了括号的方式标志
        # 为了避免引入额外的噪音，我们对sentence中含有括号部分进行了消除
        sentence1 = row['sentence1']
        sentence2 = row['sentence2']
        count1 = sentence_letter(sentence1)
        count2 = sentence_letter(sentence2)
        letter_count_list.append(count1)
        letter_count_list.append(count2)
        if count1 >= 5 or count2 >= 5:
            clean_data.append(row)
            continue  # 跳过该行
        else:
            cleaned_data.append(row)
    
    # 将 cleaned_data 转换为 DataFrame
    cleaned_df = pd.DataFrame(cleaned_data)
    clean_df = pd.DataFrame(clean_data)
    # 返回清洗后的 DataFrame
    return cleaned_df,clean_df




if __name__ == "__main__":
    # 1.数据读取
    mode = "train"
    df_train  = data_explore(mode)
    mode = "val"
    df_val = data_explore(mode)
    df =pd.concat([df_train,df_val],axis=0)

    # 2.句子长度分布展示
    print("1.原始df的形状",df.shape)
    max_length,length,max_text1,max_text2 = show_length(df,77)
    print("平均句子长度为:",np.mean(length),"个字")
    print("最短句子长度为:",min(length),"个字")
    print("最大句子长度为:",max_length,"个字")
    print("句子1:",max_text1)
    print("句子2:",max_text2)
    draw_length(length,"原始数据")

    # 3，句子类别分布展示
    draw_label(df_train,"训练数据")
    draw_label(df_val,"验证数据")
 

    # 3.数据清理部分
    df ,clean_df= data_clean(df)
    print("清理后df的形状",df.shape)

    clean_df.to_csv("clean_data.csv",index=False)
    print("去除的数据已经保存为csv文件:  clean_data.csv  ")

    # max_length,length,max_text1,max_text2 = show_length(df)
    # print("最大句子长度为:",max_length,"个字")
    # print("句子1:",max_text1)
    # print("句子2:",max_text2)
    # draw_length(length)

