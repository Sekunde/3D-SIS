import argparse
import torch
import os
import pprint

from lib.utils.config import cfg
from lib.utils.config import cfg_from_file
from lib.utils.config import cfg_to_file
from lib.model.trainval import train
from lib.model.trainval import test
from lib.model.trainval import benchmark
from lib.datasets.dataset import Dataset

def parse_args():
    """parse input arguments"""
    parser = argparse.ArgumentParser(description='3D-SIS')
    parser.add_argument('--output_dir', type=str, default='../checkpoints/')
    parser.add_argument('--epochs', help='number of epochs to train', default=100000, type=int)
    parser.add_argument('--cfg', type=str, help='optional config file', default='classification_rpnloss')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--step', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # ------------------------------------
    # args
    # ------------------------------------
    args = parse_args()
    print('Called with args:')
    print(args)

    # ------------------------------------
    # cfg
    # ------------------------------------
    if args.cfg is not None:
        cfg_from_file(os.path.join('experiments', 'cfgs', args.cfg + '.yml'))
        cfg.DEBUG = args.debug
        if cfg.LABEL_MAP != '':
            _, weights = Dataset.load_mapping(cfg.LABEL_MAP)
            cfg.NORMALIZE_WEIGHTS = []
            for weight in weights:
                if weight > 0:
                    cfg.NORMALIZE_WEIGHTS.append(weight)
            cfg.NUM_CLASSES = len(cfg.NORMALIZE_WEIGHTS)
    print('Using configs:')
    pprint.pprint(cfg)

    # ------------------------------------
    # gpu
    # ------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_num_threads(args.num_workers)

    # ------------------------------------
    # experiment dir
    # ------------------------------------
    args.output_dir = os.path.join(args.output_dir,
                                   '{}-{}'.format(args.cfg, args.tag) if args.tag is not None
                                   else args.cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    args.cfg = args.cfg.split('/')[-1]
    if not os.path.isfile(os.path.join(args.output_dir, args.cfg + '.yml')):
        cfg_to_file(os.path.join(args.output_dir, args.cfg + '.yml'))

    # ------------------------------------
    # train or eval
    # ------------------------------------
    if args.mode == 'train':
        cfg.MODE = 'train'
        train(args)
    elif args.mode == 'test':
        cfg.MODE = 'test'
        test(args)
    elif args.mode == 'benchmark':
        cfg.MODE = 'benchmark'
        benchmark(args)
