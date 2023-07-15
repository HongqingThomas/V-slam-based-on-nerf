import argparse
import random

import numpy as np
import torch

from src import config
from src.NERF_SLAM import NERF_SLAM


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
        description='Arguments for running the SLAM system.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/base_config.yaml')

    slam = NERF_SLAM(cfg, args)

    slam.run()


if __name__ == '__main__':
    main()
