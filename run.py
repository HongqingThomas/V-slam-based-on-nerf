import argparse
import random

import numpy as np
import torch

from src import config
from src.NICE_SLAM import NICE_SLAM


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # setup_seed(20)
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')

    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    parser.set_defaults(nice=True)
    args = parser.parse_args()

    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')

    slam = NICE_SLAM(cfg, args)

    slam.run()


if __name__ == '__main__':
    main()
