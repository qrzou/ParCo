import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import clip


from dataset import dataset_TM_eval_bodypart

import models.vqvae_bodypart as vqvae_bodypart
import models.t2m_trans_bodypart as trans_bodypart

from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper

import utils.utils_model as utils_model
import utils.eval_bodypart as eval_bodypart
from utils.word_vectorizer import WordVectorizer
from utils.misc import EasyDict

import argparse
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Evaluate the real motion',
                                 add_help=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eval-exp-dir', type=str, help='The trained transformer experiment directory to be evaluated.')
parser.add_argument("--skip-mmod", action='store_true', help="Skip evaluating MultiModality")
parser.add_argument('--select-ckpt', type=str, help='Select which ckpt for use: [last, fid, div, top1, matching]')
parser.add_argument("--fixed-seed", action='store_true', help="Use the same seed used in training, otherwise set the seed randomly")
parser.add_argument("--skip-path-check", action='store_true', help="Skip check of path consistency")


test_args = parser.parse_args()

assert test_args.select_ckpt in ['last', 'fid', 'div', 'top1', 'matching']

eval_exp_dir = test_args.eval_exp_dir
select_ckpt = test_args.select_ckpt
skip_mmod = test_args.skip_mmod
fixed_seed = test_args.fixed_seed
skip_path_check = test_args.skip_path_check

if skip_mmod:
    print('\n\nSkip evaluating MultiModality\n\n')
else:
    print('\n\nEvaluate MultiModality 5 times. (MDM: 5 times. T2M and T2M-GPT: 20 times). \n\n')


assert select_ckpt in [
    'last',  # last  saved ckpt
    'fid',  # best FID ckpt
    'div',  # best diversity ckpt
    'top1',  # best top-1 R-precision
    'matching',  # MM-Dist: Multimodal Distance
]


trans_config_path = os.path.join(eval_exp_dir, 'train_config.json')


# Checkpoint path
if select_ckpt == 'last':
    trans_ckpt_path = os.path.join(eval_exp_dir, 'net_' + select_ckpt + '.pth')
else:
    trans_ckpt_path = os.path.join(eval_exp_dir, 'net_best_' + select_ckpt + '.pth')



with open(trans_config_path, 'r') as f:
    trans_config_dict = json.load(f)  # dict
args = EasyDict(trans_config_dict)

vqvae_train_args = EasyDict(args.vqvae_train_args)

if fixed_seed:
    torch.manual_seed(args.seed)
    test_args.seed = args.seed
else:
    random_seed = torch.randint(0,256,[])
    test_args.seed = int(random_seed)
    print('random_seed:', random_seed)
    torch.manual_seed(random_seed)


if skip_path_check:
    print('\n Skip check of path consistency\n')
else:
    print('\n Checking path consistency...\n')
    print(eval_exp_dir)
    print(args.run_dir)
    assert os.path.samefile(eval_exp_dir, args.run_dir)

test_out_dir = os.path.join(eval_exp_dir, 'test_trans-' + select_ckpt)
test_npy_save_dir = os.path.join(test_out_dir, 'saved_npy')

os.makedirs(test_out_dir, exist_ok = True)
os.makedirs(test_npy_save_dir, exist_ok = True)



##### ---- Logger ---- #####
test_args.trans_training_config = trans_config_dict
test_args.test_out_dir = test_out_dir
test_args.test_npy_save_dir = test_npy_save_dir

logger = utils_model.get_logger(test_out_dir)
writer = SummaryWriter(test_out_dir)

logger.info(json.dumps(vars(test_args), indent=4, sort_keys=True))


# save the training config
test_args.args_save_dir = os.path.join(test_out_dir, 'train_config.json')
args_dict = vars(test_args)
with open(test_args.args_save_dir, 'wt') as f:
    json.dump(args_dict, f, indent=4)




w_vectorizer = WordVectorizer('./glove', 'our_vab')

val_loader = dataset_TM_eval_bodypart.DATALoader(args.dataname, True, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####

## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False



print('constructing vqvae')
net = vqvae_bodypart.HumanVQVAEBodyPart(
    vqvae_train_args,  # use args to define different parameters in different quantizers
    vqvae_train_args['vqvae_arch_cfg']['parts_code_nb'],
    vqvae_train_args['vqvae_arch_cfg']['parts_code_dim'],
    vqvae_train_args['vqvae_arch_cfg']['parts_output_dim'],
    vqvae_train_args['vqvae_arch_cfg']['parts_hidden_dim'],
    vqvae_train_args['down_t'],
    vqvae_train_args['stride_t'],
    vqvae_train_args['depth'],
    vqvae_train_args['dilation_growth_rate'],
    vqvae_train_args['vq_act'],
    vqvae_train_args['vq_norm']
)

print('loading checkpoint from {}'.format(args.vqvae_ckpt_path))
ckpt = torch.load(args.vqvae_ckpt_path, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()



trans_encoder = trans_bodypart.TransformerFuseHiddenDim(

    clip_dim=args.clip_dim,
    block_size=args.block_size,
    num_layers=args.num_layers,
    n_head=args.n_head_gpt,
    drop_out_rate=args.drop_out_rate,
    fc_rate=args.ff_rate,

    # FusionModule
    use_fuse=args.use_fuse,
    fuse_ver=args.fuse_ver,
    alpha=args.alpha,

    parts_code_nb=args.trans_arch_cfg['parts_code_nb'],
    parts_embed_dim=args.trans_arch_cfg['parts_embed_dim'],
    num_mlp_layers=args.trans_arch_cfg['num_mlp_layers'],
    fusev2_sub_mlp_out_features=args.trans_arch_cfg['fusev2_sub_mlp_out_features'],
    fusev2_sub_mlp_num_layers=args.trans_arch_cfg['fusev2_sub_mlp_num_layers'],
    fusev2_head_mlp_num_layers=args.trans_arch_cfg['fusev2_head_mlp_num_layers'],

)


print('loading transformer checkpoint from {}'.format(trans_ckpt_path))
ckpt = torch.load(trans_ckpt_path, map_location='cpu')
trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()

print('Done')

fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
multi = []
repeat_time = 20

'''
mmod_times = 20  # T2M, T2M-GPT and some other papers evaluate multimodality 20 times
mmod_times = 5  # MDM evaluate multimodality 5 times
mmod_times = 0  # no multimodality evaluation
'''

if skip_mmod:
    mmod_times = 0
else:
    mmod_times = 5


for i in range(repeat_time):

    if i < mmod_times:  # 0~4
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_bodypart.evaluation_transformer_test_batch(test_npy_save_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_multi=0, clip_model=clip_model, eval_wrapper=eval_wrapper, draw=False, savegif=False, save=False, savenpy=False, mmod_gen_times=30, skip_mmod=False)
        multi.append(best_multi)
    else:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_bodypart.evaluation_transformer_test_batch(test_npy_save_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_multi=0, clip_model=clip_model, eval_wrapper=eval_wrapper, draw=False, savegif=False, save=False, savenpy=False, mmod_gen_times=1, skip_mmod=True)

    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)


print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)

if mmod_times == 0:
    pass
else:
    print('multi: ', sum(multi)/mmod_times)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
if mmod_times == 0:
    msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, Multi. {0:.3f}, conf. {0:.3f}"
else:
    multi = np.array(multi)
    msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, Multi. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(mmod_times):.3f}"
logger.info(msg_final)