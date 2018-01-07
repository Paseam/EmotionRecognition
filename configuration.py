# coding:utf8
class DefaultConfig(object):
    env = 'main' # visdom 环境
    model = 'AlexNet' # 使用的模型，名字必须与models/__init__.py中的名字一致
    num_classes = 7
    
    train_data_root = '/home/lc/FaceExpression/train/KDEF_10G/face/' # 训练集存放路径
    test_data_root = '/home/lc/FaceExpression/validation/face/' # 测试集存放路径
    pretrained_model_path = None # './checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载

    batch_size = 64 # batch size
    use_gpu = True # user GPU or not
    num_workers = 4 # how many workers for loading data
    print_freq = 20 # print info every N batch

    debug_file = './debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
      
    max_epoch = 300
    lr = 0.1 # initial learning rate
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # 损失函数

config = DefaultConfig()
