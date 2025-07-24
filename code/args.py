import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='机器学习项目参数配置')
    
    # 数据参数
    parser.add_argument('--train_path', type=str, default='/hy-tmp/dataset/train/train.csv', help='训练数据路径')
    parser.add_argument('--val_path', type=str, default='/hy-tmp/dataset/val/val.csv', help='验证数据路径')
    parser.add_argument('--file_mode', type=str, default='csv', help='验证数据路径')
    # 模型参数
    parser.add_argument('--model_name', type=str, default='bert-base-chinese',help='使用的模型名称')
    parser.add_argument('--pretrained_path', type=str, default='/hy-tmp/code/bert-base-chinese',help='预训练模型路径')
    parser.add_argument('--save_path', type=str, default='/hy-tmp/model_pth/', help='验证数据路径')
    parser.add_argument('--num_classes', type=int, default=3,help='分类数量')
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=192,help='训练批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5,help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-3,help='L2正则化率')
    parser.add_argument('--epochs', type=int, default=10,help='训练轮数')
    parser.add_argument('--loss_function',type=str,default="CrossEntropy",
                        choices = ["CrossEntropy","CEFL","CEFL2"],help='损失函数选择')

    parser.add_argument('--NCE_function',type=str,default=False,help='NCE函数开启')

    parser.add_argument('--augment', type=str, default=True,choices=[False,True],help='数据增强开启/关闭')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('参数配置:')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')