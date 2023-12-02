import argparse
import os
import pathlib
import re
import time
import datetime
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import collections

import logging
import datetime
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from dataset import build_poisoned_training_set, build_testset
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch
from models import BadNet
from models.resnet_with_adapter import resnet18,freeze_model_except_adapters
from models.resnet18 import buildresnet18

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get the current date and time
now = datetime.datetime.now()

log_dir = now.strftime('tensorboardlogs/%Y-%m-%d_%H-%M-%S')

# Create a SummaryWriter
writer = SummaryWriter(log_dir)

# Format the date and time as a string
filename = now.strftime('logs/train_%Y-%m-%d_%H-%M-%S.log')

# Create a file handler
handler = logging.FileHandler(filename)
handler.setLevel(logging.INFO)

# Format the date and time as a string
sub_dir = now.strftime('%Y-%m-%d_%H-%M-%S')

# Create the subdirectory if it does not exist
os.makedirs(os.path.join("Trained models", sub_dir), exist_ok=True)

# Create a stream handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter('[%(asctime)s] - [Model] - [Dataset] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Add the formatter to the handler
handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)
logger.addHandler(stream_handler)



parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=8, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cpu', help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')
parser.add_argument('--pretrained', type=str,default='./resnet18-f37072fd.pth',action='store_true', help='Do you want to use pretrained model ( default false, if you add this param, then use pretrained model)')
#seed
parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 3407)')


args = parser.parse_args()

def main():
    seed_everything(args.seed)
    logger.info("{}".format(args).replace(', ', ',\n'))

    if re.match('cuda:\d', args.device):
        cuda_num = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if you're using MBP M1, you can also use "mps"

    # create related path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    logger.info("\n# load dataset: %s " % args.dataset)
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)
    
    data_loader_train        = DataLoader(dataset_train,         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) # shuffle 随机化

    # model = BadNet(input_channels=dataset_train.channels, output_num=args.nb_classes).to(device)
    model=resnet18(pretrained=False, num_classes=args.nb_classes).to(device)
    freeze_model_except_adapters(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_picker(args.optimizer, model.parameters(), lr=args.lr)

    basic_model_path = "./checkpoints/resnet18-%s.pth" % args.dataset
    start_time = time.time()
    if args.pretrained:
        pretrained_dict = torch.load(args.pretrained)
        pretrained_dict=collections.OrderedDict(pretrained_dict)
        backbone_dict = {}
        for k in list(pretrained_dict.keys()):
            if (k.startswith('conv1')) or (k.startswith('bn1')) or (k.startswith('layer')):
                backbone_dict[k.replace('', '')] = pretrained_dict[k]
        result=model.load_state_dict(backbone_dict, strict=False)
        logger.info("missing keys:{}".format(result.missing_keys))
        logger.info("unexpected keys:{}".format(result.unexpected_keys))
    elif args.load_local:
        logger.info("## Load model from : %s" % basic_model_path)
        model.load_state_dict(torch.load(basic_model_path), strict=True)
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
        logger.info(f"Test Clean Accuracy(TCA): {test_stats['clean_acc']:.4f}")
        logger.info(f"Attack Success Rate(ASR): {test_stats['asr']:.4f}")
    else:
        logger.info(f"Start training for {args.epochs} epochs")
        stats = []
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(data_loader_train, model, criterion, optimizer, args.loss, device)
            test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
            logger.info(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")
            
            # save model 
            torch.save(model.state_dict(), basic_model_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
            }

            # save training stats
            stats.append(log_stats)
            df = pd.DataFrame(stats)
            df.to_csv("./logs/%s_trigger%d.csv" % (args.dataset, args.trigger_label), index=False, encoding='utf-8')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    main()
