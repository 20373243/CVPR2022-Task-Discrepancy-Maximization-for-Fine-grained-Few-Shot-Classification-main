import os
import sys
import torch
import torch.optim as optim
import logging
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .eval import meta_test
sys.path.append('..')
from datasets import dataloaders


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_logger(filename):

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",datefmt='%m/%d %I:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def train_parser():
    parser = argparse.ArgumentParser()

    ## general hyper-parameters
    parser.add_argument("--opt", help="optimizer", choices=['adam','sgd'], default='sgd')
    parser.add_argument("--lr", help="initial learning rate", type=float, default=0.1)
    parser.add_argument("--gamma", help="learning rate cut scalar", type=float, default=0.1)
    parser.add_argument("--epoch", help="number of epochs before lr is cut by gamma", type=int, default=400)
    parser.add_argument("--stage", help="number lr stages", type=int)#add default
    parser.add_argument("--weight_decay", help="weight decay for optimizer", type=float, default=5e-4)
    parser.add_argument("--gpu_num", help="gpu device", type=int, default=1)
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument("--val_epoch", help="number of epochs before eval on val", type=int, default=20)
    parser.add_argument("--resnet", help="whether use resnet12 as backbone or not", action="store_true")
    parser.add_argument("--nesterov", help="nesterov for sgd", action="store_true")#
    parser.add_argument("--batch_size", help="batch size used during pre-training", type=int)
    parser.add_argument('--decay_epoch', nargs='+',help='epochs that cut lr', type=int)
    parser.add_argument("--pre", help="whether use pre-resized 84x84 images for val and test", action="store_true")
    parser.add_argument("--no_val", help="don't use validation set, just save model at final timestep", action="store_true")
    parser.add_argument("--train_way", help="training way", type=int)# add
    parser.add_argument("--test_way", help="test way", type=int, default=2)
    parser.add_argument("--train_shot", help="number of support images per class for meta-training and meta-testing during validation", type=int)# add
    parser.add_argument("--test_shot", nargs='+', help="number of support images per class for meta-testing during final test", type=int)#add
    parser.add_argument("--train_query_shot", help="number of query images per class during meta-training", type=int, default=15)
    parser.add_argument("--test_query_shot", help="number of query images per class during meta-testing", type=int, default=16)
    parser.add_argument("--train_transform_type", help="size transformation type during training", type=int)#add
    parser.add_argument("--test_transform_type", help="size transformation type during inference", type=int)
    parser.add_argument("--val_trial", help="number of meta-testing episodes during validation", type=int, default=1000)
    parser.add_argument("--detailed_name", help="whether include training details in the name", action="store_true")

    parser.add_argument("--model", choices=['Proto', 'FRN'])
    parser.add_argument("--dataset", choices=['cub_cropped', 'cub_raw',
                                              'aircraft',
                                              'meta_iNat', 'tiered_meta_iNat',
                                              'stanford_car', 'stanford_dog','medical'])
    parser.add_argument("--TDM", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--noise_value", type=float, default=0.2)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_epoch", type=int, default=0)

    args = parser.parse_args()

    return args


def get_opt(model, args):

    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
       # params(iterable)：可用于迭代优化的参数或者定义参数组的dicts。
       # lr(float, optional) ：学习率(默认: 1e-3)
       # betas(Tuple[float, float], optional)：用于计算梯度的平均和平方的系数(默认: (0.9, 0.999))
       # eps(float, optional)：为了提高数值稳定性而添加到分母的一个项(默认: 1e-8)
       # weight_decay(float, optional)：权重衰减(如L2惩罚)(默认: 0)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9,
                              weight_decay=args.weight_decay,
                              nesterov=args.nesterov)

    if args.decay_epoch is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.decay_epoch, gamma=args.gamma)
        #动态调整学习率的函数
        #1）milestones为一个数组，如 [50,70]；
        #2）gamma为倍数，如果learning rate开始为0.01 ，则当epoch为50时变为0.001，epoch 为70 时变为0.0001。
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epoch, gamma=args.gamma)

    return optimizer, scheduler


class Path_Manager:
    def __init__(self, fewshot_path, args):
        self.train = os.path.join(fewshot_path,'train')

        if args.pre:
            self.test = os.path.join(fewshot_path,'test_pre')
            self.val = os.path.join(fewshot_path,'val_pre') if not args.no_val else self.test

        else:
            self.test = os.path.join(fewshot_path,'test')
            self.val = os.path.join(fewshot_path,'val') if not args.no_val else self.test


class Train_Manager:
    def __init__(self, args, path_manager, train_func):

        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        #设置固定生成随机数的种子，使得每次运行该.py文件时生成的随机数相同
        np.random.seed(seed)
        #使每次随机生成数一样

        if args.resnet:
            name = 'ResNet-12'
        else:
            name = 'Conv-4'

        if args.detailed_name:
            if args.decay_epoch is not None:
                temp = ''
                for i in args.decay_epoch:
                    temp += ('_'+str(i))

                suffix = '%s-lr_%.0e-gamma_%.0e-epoch_%d-drop%s-decay_%.0e-way_%d' % \
                         (args.opt, args.lr, args.gamma, args.epoch, temp, args.weight_decay, args.train_way)
            else:
                suffix = '%s-lr_%.0e-gamma_%.0e-epoch_%d-stage_%d-decay_%.0e-way_%d' % \
                         (args.opt, args.lr, args.gamma, args.epoch, args.stage, args.weight_decay, args.train_way)

            name = "%s-%s" % (name, suffix)

        check_dir(args.save_folder)
        if args.resume:
            file = open(os.path.join(args.save_folder, '%s.log' % (name)), 'r')
            lines = file.read().splitlines()
            file.close()

        self.logger = get_logger(os.path.join(args.save_folder, '%s.log' % (name)))
        self.save_path = os.path.join(args.save_folder, 'model_%s.pth' % (name))
        self.writer = SummaryWriter(os.path.join(args.save_folder, 'log_%s' % (name)))

        if args.resume:
            self.logger.info('display resume information')
            for i in range(len(lines)):
                self.logger.info(lines[i][17:])
            self.logger.info('--------------------------')

        self.logger.info('display all the hyper-parameters in args:')
        for arg in vars(args):
            value = getattr(args, arg)
            if value is not None:
                self.logger.info('%s: %s' % (str(arg), str(value)))
        self.logger.info('------------------------')
        self.args = args
        self.train_func = train_func
        self.pm = path_manager

    def train(self, model):

        args = self.args
        train_func = self.train_func
        writer = self.writer
        save_path = self.save_path
        logger = self.logger

        optimizer, scheduler = get_opt(model, args)
        #优化器、动态调整学习参数

        val_shot = args.train_shot
        test_way = args.test_way

        best_val_acc = 0
        best_epoch = 0

        model.train()
        #model.train()的作用是启用 Batch Normalization 和 Dropout。
        #如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
        #model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
        model.cuda()
        #调用model.cuda()，可以将模型加载到GPU上去

        if args.gpu_num > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.gpu_num)))
            #module即表示你定义的模型；device_ids表示你训练的device；
            #当迭代次数或者epoch足够大的时候，使用nn.DataParallel函数来用多个GPU来加速训练。

        if args.decay_epoch is not None:
            total_epoch = args.epoch
        else:
            total_epoch = args.epoch*args.stage

        logger.info("start training!")

        iter_counter = 0
        for e in tqdm(range(total_epoch)):
            if (args.resume + 1) and args.resume_epoch > (e + 1):
                pass
            else:
                iter_counter, train_acc = train_func(model=model,
                                                     optimizer=optimizer,
                                                     writer=writer,
                                                     iter_counter=iter_counter,
                                                     gpu_num=args.gpu_num)

                if (e+1) % args.val_epoch == 0:

                    logger.info("")
                    logger.info("epoch %d/%d, iter %d:" % (e+1,total_epoch,iter_counter))
                    logger.info("train_acc: %.3f" % (train_acc))

                    model.eval()
                    #在模型预测阶段，我们需要将这些层设置到预测模式
                    with torch.no_grad():
                        #在预测阶段，也会加上torch.no_grad()来关闭梯度的计
                        val_acc, val_interval = meta_test(data_path=self.pm.val,
                                                          model=model,
                                                          way=test_way,
                                                          shot=val_shot,
                                                          pre=args.pre,
                                                          transform_type=args.test_transform_type,
                                                          query_shot=args.test_query_shot,
                                                          trial=args.val_trial,
                                                          gpu_num=args.gpu_num
                                                          )
                        writer.add_scalar('val_%d-way-%d-shot_acc' % (test_way, val_shot), val_acc, iter_counter)

                    logger.info('val_%d-way-%d-shot_acc: %.3f\t%.3f' % (test_way, val_shot, val_acc, val_interval))

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = e+1
                        if not args.no_val:
                            if args.gpu_num > 1:
                                torch.save(model.module.state_dict(), save_path)
                            else:
                                torch.save(model.state_dict(), save_path)
                        logger.info('BEST!')

                    model.train()

            scheduler.step()

        logger.info('training finished!')
        if args.no_val:
            torch.save(model.state_dict(), save_path)

        logger.info('------------------------')
        logger.info(('the best epoch is %d/%d') % (best_epoch,total_epoch))
        logger.info(('the best %d-way %d-shot val acc is %.3f') % (test_way,val_shot,best_val_acc))

    def evaluate(self, model):

        logger = self.logger
        args = self.args

        logger.info('------------------------')
        logger.info('evaluating on test set:')

        with torch.no_grad():

            try:
                model.load_state_dict(torch.load(self.save_path))
            except:
                model.module.load_state_dict(torch.load(self.save_path))
            model.eval()

            for shot in args.test_shot:

                mean, interval = meta_test(data_path=self.pm.test,
                                           model=model,
                                           way=args.test_way,
                                           shot=shot,
                                           pre=args.pre,
                                           transform_type=args.test_transform_type,
                                           query_shot=args.test_query_shot,
                                           trial=1000,
                                           gpu_num=args.gpu_num
                                           )

                logger.info('%d-way-%d-shot acc: %.2f\t%.2f'%(args.test_way,shot,mean,interval))

