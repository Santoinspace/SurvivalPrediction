import argparse

def get_args():
    model_name = 'model_1'
    parser = argparse.ArgumentParser(description='MultiSurv 模型配置参数')

    # 路径设置
    path_group = parser.add_argument_group('路径设置')
    path_group.add_argument('--results_path', default='D:\\AProjection\\SurvivalPrediction\\results',
                            help='结果保存路径')
    path_group.add_argument('--data_path', default='D:\\AProjection\\SurvivalPrediction\\data\\preprocessed', 
                            help='数据目录路径')
    path_group.add_argument('--tabular_name', default='clinal_test.csv', 
                            help='临床数据文件名')
    path_group.add_argument('--model_path', default=f'D:\\AProjection\\SurvivalPrediction\\results\\{model_name}\\checkpoints',
                            help='模型保存路径')

    # 模型参数
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument('--model_name', default=model_name,
                           help='模型名称')
    model_group.add_argument('--t_dim', type=int, default=21, 
                           help='表格数据的特征维度')
    model_group.add_argument('--interval_num', type=int, default=4,
                           help='生存区间数量')
    model_group.add_argument('--train_ratio', type=float, default=0.2,
                           help='训练集比例')
    model_group.add_argument('--val_ratio', type=float, default=0.8,
                           help='验证集比例')
    model_group.add_argument('--test_ratio', type=float, default=0,
                           help='测试集比例')

    # 时间区间配置
    time_group = parser.add_argument_group('时间区间配置')
    time_group.add_argument('--intervals', nargs='+', type=int, 
                          default=[0, 15, 30, 45, 60],
                          help='生存时间区间分割点')

    # 训练配置
    train_group = parser.add_argument_group('训练配置')
    train_group.add_argument('--mode', default='train', choices=['train', 'test'],
                           help='运行模式：训练或测试')
    train_group.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                           help='使用的GPU设备编号')
    train_group.add_argument('--seed', type=int, default=0,
                           help='随机种子')
    train_group.add_argument('--fold_num', type=int, default=5,
                           help='交叉验证折数')
    train_group.add_argument('--epoch_num', type=int, default=10,
                           help='训练总轮次')
    train_group.add_argument('--epoch_save_model_interval', type=int, default=5,
                           help='保存模型的间隔轮次')
    train_group.add_argument('--lr', type=float, default=1,
                             help='学习率')
    train_group.add_argument('--weight_decay', type=float, default=0.0001,
                             help='权重衰减')
    train_group.add_argument('--momentum', type=float, default=0.9,
                             help='动量')
    train_group.add_argument('--batch_size', type=int, default=2,
                           help='训练批大小')
    train_group.add_argument('--batch_size_eval', type=int, default=1,
                           help='评估批大小')
    train_group.add_argument('--num_workers', type=int, default=0,
                           help='数据加载线程数')
    train_group.add_argument('--pin_memory', action='store_true',
                            help='是否使用pin_memory')
    train_group.add_argument('--resume', type=bool, default=False,
                            help='是否从断点继续训练')

    # 可视化配置
    vis_group = parser.add_argument_group('可视化配置')
    vis_group.add_argument('--color_train', default='#f14461',
                         help='训练曲线颜色')
    vis_group.add_argument('--color_eval', default='#3498db',
                         help='验证曲线颜色')
    vis_group.add_argument('--color_test', default='#27ce82',
                         help='测试曲线颜色')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print(args.color_test)
    # print(args)
