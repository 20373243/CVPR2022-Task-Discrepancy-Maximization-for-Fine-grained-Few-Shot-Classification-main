import os
import math
import torch
import torchvision.datasets as datasets
import numpy as np
from copy import deepcopy
from PIL import Image
from . import samplers, transform_manager


def get_dataset(data_path, is_training, transform_type, pre):
    dataset = datasets.ImageFolder(
        data_path,
        loader=lambda x: image_loader(path=x, is_training=is_training, transform_type=transform_type, pre=pre))
    #path：图片存储的根目录
    #loader：表示数据集加载方式

    return dataset



def meta_train_dataloader(data_path,way,shots,transform_type):

    dataset = get_dataset(data_path=data_path,is_training=True,transform_type=transform_type,pre=None)

    loader = torch.utils.data.DataLoader(
    #从dataset数据库中每次抽出batch_size个数据
        dataset,
        batch_sampler=samplers.meta_batchsampler(data_source=dataset, way=way, shots=shots),
        num_workers=0,
        pin_memory=False)
    #batch_sampler 是一个迭代器， 每次生次一个batch_size的key用于读取dataset中的值
    #num_workers 参与工作的线程数

    return loader



def meta_test_dataloader(data_path,way,shot,pre,transform_type=None,query_shot=16,trial=1000):

    dataset = get_dataset(data_path=data_path,is_training=False,transform_type=transform_type,pre=pre)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=samplers.random_sampler(data_source=dataset,way=way,shot=shot,query_shot=query_shot,trial=trial),
        num_workers=0,
        pin_memory=False)

    return loader


def normal_train_dataloader(data_path,batch_size,transform_type):

    dataset = get_dataset(data_path=data_path,
                          is_training=True,
                          transform_type=transform_type,
                          pre=None)

    loader = torch.utils.data.DataLoader(
    #从dataset数据库中每次抽出batch_size个数据
        dataset,
        batch_size=4, #add
        #batch_size=batch_size,
        shuffle=True,#将数据打乱
        num_workers=0,#使用8个线程
        pin_memory=False,
        drop_last=True)
    #drop_last 对最后不足batchsize的数据的处理方法

    return loader


def image_loader(path, is_training, transform_type,pre):

    p = Image.open(path)
    #利用img = Image.open(ImgPath) 打开的图片是PIL类型的
    p = p.convert('RGB')
    #转换成RGB模式

    final_transform = transform_manager.get_transform(is_training=is_training,
                                                      transform_type=transform_type,
                                                      pre=pre)

    p = final_transform(p)

    return p
