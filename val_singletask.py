import os
import torch
import torch.optim.lr_scheduler
import torch.backends.cudnn as cudnn
import yaml
import math
from copy import deepcopy
from argparse import ArgumentParser

from model.model import SingleLiteNetPlus
from loss import SigleLoss
from utils import train, val_one, netParams, save_checkpoint, poly_lr_scheduler
import BDD100K

def val_net(args, hyp):
    """Train the neural network model with given arguments and hyperparameters"""
    cuda_available = torch.cuda.is_available()
    
    model = SingleLiteNetPlus(args)
    # if num_gpus > 1:
    #     model = torch.nn.DataParallel(model)
    
    valLoader = torch.utils.data.DataLoader(
        BDD100K.DatasetOneTask(hyp, valid=True, task=args.task, width=args.width, height=args.height),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True
    
    print(f'Total network parameters: {netParams(model)}')
    
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    
    # Perform validation
    segment_results = val_one(valLoader, model, args=args)
    
    # Print results
    print(f"Driving Area Segment: Acc({segment_results[0]:.3f}) IOU({segment_results[1]:.3f}) mIOU({segment_results[2]:.3f})") if args.task=="DA" else \
            print(f"Lane Line Segment: Acc({segment_results[0]:.3f}) IOU({segment_results[1]:.3f}) mIOU({segment_results[2]:.3f})")
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=12, help='Number of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--width', type=int, default=640, help='Input image width')
    parser.add_argument('--height', type=int, default=384, help='Input image height')
    parser.add_argument('--weight', type=str, default="pretrained/large.pth", help='Path to model weights')
    parser.add_argument('--task', type=str, choices=["DA", "LL"], default=None, help="DA for drivable area, LL for lane line")
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinlitev2_hyper.yaml', help='Path to hyperparameters YAML')
    parser.add_argument('--config', default='nano', help='Model configuration')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # Load hyperparameters
    
    val_net(args, hyp.copy())
