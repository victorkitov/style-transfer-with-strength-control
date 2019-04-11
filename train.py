from pylab import *
from functions import *
from models import *
import argparse
import torch
import numpy as np
import os

parser = argparse.ArgumentParser(description='Real-time style transfer with strength control: train model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--max_train_count', default=160000, type=positive_int, help='training will stop after passing this number of images')
parser.add_argument('--log_batches_interval', default=80,  type=positive_int, help='number of batches after which the training loss is logged')
parser.add_argument('--style_image', default='images/styles/la_muse.jpg', help='path to style-image')
parser.add_argument('--gpu_id', default=0, type=int, help='GPU to use')

parser.add_argument('--style_weight', default=100000, type=float, help='weighting factor for style loss')
parser.add_argument('--tv_weight', default=0.00001, type=float, help='weighting factor for total variation loss')
parser.add_argument('--max_style_strength', default=10, type=non_negative_float, help='during training style_strength will be sampled randomly from [0,style_strength_step,...max_style_strength]')
parser.add_argument('--style_strength_step', default=0.1, type=non_negative_float, help='during training style_strength will be sampled randomly from [0,style_strength_step,...max_style_strength]')

parser.add_argument('--dataset', default='../../Datasets/Contents/MS-COCO/train2014', help='path to content images dataset on which model will be trained, should point to a folder, containing another folder with images')
parser.add_argument('--checkpoint_batches_interval', default=None, help='number of batches after which a checkpoint of the trained model will be created')
parser.add_argument('--max_style_pixels', default=400*400, type=positive_int, help='max size in total pixels count of style-image during training, None for no scaling')
parser.add_argument('--use_parallel_gpu', default=False, type=bool, help='model trained using single GPU or using parallelization over multiple GPUs')
parser.add_argument('--image_size', default=256, type=positive_int, help='during training content images are resized to this size along X and Y axis')
parser.add_argument('--batch_size', default=12, type=positive_int, help='size of batches during training')
parser.add_argument('--lr', default=0.001, type=non_negative_float, help='learning rate')
parser.add_argument('--init_model', default='', help='path to model if need model finetuning')
parser.add_argument('--save_model_dir', default='models/', help='path where model will be saved')
parser.add_argument('--checkpoint_model_dir', default='intermediate_models/', help='path to folder where checkpoints of trained models will be saved')
parser.add_argument('--seed', default=1, type=positive_int, help='random seed')
parser.add_argument('--loss_averaging_window', default=500,  type=positive_int, help='window averaging for losses (this average is displayed during training)')

args = parser.parse_args()

args.device=torch.device(f'cuda:{args.gpu_id}')
args.style_strength_grid = np.arange(0,args.max_style_strength+args.style_strength_step,args.style_strength_step)


stylizer = init_model(args)
stylizer = train(stylizer, args)
style_file = os.path.basename(args.style_image)
save_model(stylizer, args.save_model_dir+f'{style_file}_{args.max_train_count}.pth')


