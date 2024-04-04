import os
import json
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import models.vqvae_uplow as vqvae
from models.evaluator_wrapper import EvaluatorModelWrapper
from dataset import dataset_TM_eval_uplow

import options.option_vq_uplow as option_vq
from options.get_eval_option import get_opt

import utils.utils_model as utils_model
import utils.eval_uplow as eval_uplow
from utils.word_vectorizer import WordVectorizer
from utils.misc import EasyDict

# import warnings
# warnings.filterwarnings('ignore')



##### ---- Exp dirs ---- #####
test_args = option_vq.get_vavae_test_args_parser()
select_ckpt = test_args.select_vqvae_ckpt
assert select_ckpt in [
    'last',  # last  saved ckpt
    'fid',  # best FID ckpt
    'div',  # best diversity ckpt
    'top1',  # best top-1 R-precision
    'matching',  # MM-Dist: Multimodal Distance
]


vqvae_train_dir = test_args.vqvae_train_dir

# Checkpoint path
if select_ckpt == 'last':
    test_args.ckpt_path = os.path.join(vqvae_train_dir, 'net_' + select_ckpt + '.pth')
else:
    test_args.ckpt_path = os.path.join(vqvae_train_dir, 'net_best_' + select_ckpt + '.pth')

# Prepare testing directory
test_args.test_dir = os.path.join(vqvae_train_dir, 'test_vqvae-' + select_ckpt)
test_args.test_npy_save_dir = os.path.join(test_args.test_dir, 'saved_npy')
os.makedirs(test_args.test_dir, exist_ok=True)
os.makedirs(test_args.test_npy_save_dir, exist_ok=True)

# Load the config of vqvae training
print('\nLoading training argument...\n')
test_args.training_options_path = os.path.join(vqvae_train_dir, 'train_config.json')
with open(test_args.training_options_path, 'r') as f:
    train_args_dict = json.load(f)  # dict
train_args = EasyDict(train_args_dict)  # convert dict to easydict for convenience
test_args.train_args = train_args  # save train_args into test_args for logging convenience


##### ---- Logger ---- #####
logger = utils_model.get_logger(test_args.test_dir )
writer = SummaryWriter(test_args.test_dir )
logger.info(json.dumps(vars(test_args), indent=4, sort_keys=True))


w_vectorizer = WordVectorizer('./glove', 'our_vab')

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if train_args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####

val_loader = dataset_TM_eval_uplow.DATALoader(
    train_args.dataname, True, 32, w_vectorizer, unit_length=2**train_args.down_t)


##### ---- Network ---- #####
print('\n\n===> Constructing network...')
net = vqvae.HumanVQVAEUpLow(
    train_args,  # use args to define different parameters in different quantizers
    train_args['vqvae_arch_cfg']['parts_code_nb'],
    train_args['vqvae_arch_cfg']['parts_code_dim'],
    train_args['vqvae_arch_cfg']['parts_output_dim'],
    train_args['vqvae_arch_cfg']['parts_hidden_dim'],
    train_args['down_t'],
    train_args['stride_t'],
    train_args['depth'],
    train_args['dilation_growth_rate'],
    train_args['vq_act'],
    train_args['vq_norm']
)


#### Loading weights #####
print('\n\n===> Loading weights...')
if test_args.ckpt_path:
    logger.info('loading checkpoint from {}'.format(test_args.ckpt_path))
    ckpt = torch.load(test_args.ckpt_path, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
else:
    raise Exception('You need to specify the ckpt path!')

net.cuda()
net.eval()

fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
repeat_time = 20
print('\n===> Start testing...')
for i in range(repeat_time):
    print('\n===> Test round:', i)
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
        eval_uplow.evaluation_vqvae(
            test_args.test_npy_save_dir, val_loader, net, logger, writer,
            0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0,
            best_matching=100, eval_wrapper=eval_wrapper, draw=False, save=False, savenpy=(i==0))

    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)

print('\n\nfinal result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)