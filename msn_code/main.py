# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#%%

import argparse

import torch.multiprocessing as mp

import pprint
import yaml

from src.msn_train import main as msn

from src.utils import init_distributed

parser = argparse.ArgumentParser()

parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='./configs/msn_vits16.yaml',)


parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')

parser.add_argument(
    '--world-size', type=int, default=1,
    help='number of distributed processes')

parser.add_argument(
    '--rank', type=int, default=0,
    help='the rank of this process')


def process_main(rank, fname, world_size, devices):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    dump = os.path.join(params['logging']['folder'], 'params-msn-train.yaml')
    with open(dump, 'w') as f:
        yaml.dump(params, f)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    return msn(params)


if __name__ == '__main__':

    args = parser.parse_args()

    num_gpus = len(args.devices)

    process_main(args.rank, args.fname, args.world_size, args.devices)
   
    # mp.spawn(
    #     process_main,
    #     nprocs=num_gpus,
    #     args=(args.fname, num_gpus, args.devices))

