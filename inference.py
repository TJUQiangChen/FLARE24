import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import torch

from utils.common_function import parse_option

from engines.common import Inference

if __name__ == "__main__":
    torch.cuda.synchronize()
    t_start = time.time()
    # get config
    _, config = parse_option("other")

    predict = Inference(config)
    predict.run()
    torch.cuda.synchronize()
    t_end = time.time()
    total_time = t_end - t_start
    print("Total_time: {} s".format(total_time))
