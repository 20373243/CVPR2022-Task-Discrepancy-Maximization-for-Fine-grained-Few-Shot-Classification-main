from functools import partial
from trainers import trainer, frn_train, proto_train
from datasets import dataloaders
from utils.util import *


args = trainer.train_parser()

args.model='FRN'
args.dataset='aircraft'
args.opt='sgd'
args.lr=0.1
args.gamma=0.1
args.epoch=400
args.stage=3
args.val_epoch=20
args.weight_decay=0.0005
args.nesterov='true'
args.train_way=5 #20
args.train_shot=5
args.train_transform_type=0
args.test_shot=15
args.pre='true'
args.gpu_num=1





assert args.gpu_num > 0, "TDM is only tested with GPU setting"

fewshot_path = dataset_path(args)

pm = trainer.Path_Manager(fewshot_path=fewshot_path, args=args)


train_loader = dataloaders.\
    meta_train_dataloader(data_path=pm.train,
                          way=args.train_way,
                          shots=[args.train_shot, args.train_query_shot],
                          transform_type=args.train_transform_type)

args.save_folder = get_save_path(args)
if args.model == 'Proto':
    train_func = partial(proto_train.default_train, train_loader=train_loader)
else:
    train_func = partial(frn_train.default_train, train_loader=train_loader)
tm = trainer.Train_Manager(args, path_manager=pm, train_func=train_func)

model = load_model(args)
if args.resume:
    model = load_resume_point(args, model)
tm.train(model)
tm.evaluate(model)