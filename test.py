from functions import *
from pylab import *
import os
import torch

import argparse

parser = argparse.ArgumentParser(description='Real-time style transfer with strength control: apply style',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--content', nargs='+', default=['images/contents/bus.jpg'], help='sequence of content images to be stylized')
parser.add_argument('--out_dir', default='images/results/', help='directory where stylized images will be stored')
parser.add_argument('--model', default='models/la_muse.pth', help='path to serialized model, obtained via train.py')
parser.add_argument('--style_strength', default=1, type=non_negative_float, help='non-negative float parameter, controlling stylization strength')
parser.add_argument('--use_parallel_gpu', default=False, type=bool, help='model trained using single GPU or using parallelization over multiple GPUs')
parser.add_argument('--gpu_id', default=0, type=int, help='GPU to use')
parser.add_argument('--scale_content', default=None, type=float, help='scaling factor for content images')
args = parser.parse_args()

args.device=torch.device(f'cuda:{args.gpu_id}')
print(args)
args.init_model = args.model
stylizer=init_model(args)
for content_file in args.content:
  print(f'Processing {content_file}...')
  result_file = f'{os.path.basename(content_file)}_{os.path.basename(args.model)}_{args.style_strength}.jpg'
  result_file = os.path.join(args.out_dir, result_file)
  result = impose_style(content_file, stylizer, args.style_strength, args)[0].cpu()
  result = tensor2image(result)
  result.save(result_file)
  print(f'Result saved to {result_file}')
