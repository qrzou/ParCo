import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import re
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
from utils.motion_process import recover_from_ric
import visualize.plot_3d_global as plot_3d

import argparse
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Infer to get the motions in .npy format, will be used by visualization',
                                 add_help=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eval-exp-dir', type=str, help='The experiment directory storing trained transformer')
parser.add_argument('--select-ckpt', type=str, help='Select which ckpt for use: [last, fid, div, top1, matching]')
parser.add_argument('--infer-mode', type=str, help='Select inferring mode: [testset, userinput]')
parser.add_argument('--input-text', type=str, help='Text for generating motion')
parser.add_argument("--fixed-seed", action='store_true', help="Use the same seed used in training, otherwise set the seed randomly")
parser.add_argument("--skip-path-check", action='store_true', help="Skip check of path consistency")

test_args = parser.parse_args()

assert test_args.select_ckpt in ['last', 'fid', 'div', 'top1', 'matching']
assert test_args.infer_mode in ['testset', 'userinput']

eval_exp_dir = test_args.eval_exp_dir
select_ckpt = test_args.select_ckpt
infer_mode = test_args.infer_mode
fixed_seed = test_args.fixed_seed
skip_path_check = test_args.skip_path_check


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



# Pick output directory.
prev_run_dirs = []
outdir = 'output/visualize'
if os.path.isdir(outdir):
    prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
cur_run_id = max(prev_run_ids, default=-1) + 1
infer_out_dir = os.path.join(outdir, f'{cur_run_id:05d}-{infer_mode}')
assert not os.path.exists(infer_out_dir)
print('Creating output directory...')
os.makedirs(infer_out_dir)



##### ---- Logger ---- #####
test_args.trans_training_config = trans_config_dict
test_args.infer_out_dir = infer_out_dir

logger = utils_model.get_logger(infer_out_dir)
writer = SummaryWriter(infer_out_dir)
logger.info(json.dumps(vars(test_args), indent=4, sort_keys=True))

# save the config
test_args.args_save_dir = os.path.join(infer_out_dir, 'infer_config.json')
args_dict = vars(test_args)
with open(test_args.args_save_dir, 'wt') as f:
    json.dump(args_dict, f, indent=4)



##### ---- Network ---- #####

## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False



print('constructing vqvae')
ParCo_vqvae = vqvae_bodypart.HumanVQVAEBodyPart(
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
ParCo_vqvae.load_state_dict(ckpt['net'], strict=True)
ParCo_vqvae.eval()
ParCo_vqvae.cuda()



ParCo_transformer = trans_bodypart.TransformerFuseHiddenDim(

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
ParCo_transformer.load_state_dict(ckpt['trans'], strict=True)
ParCo_transformer.train()
ParCo_transformer.cuda()

print('Done')




if infer_mode == 'userinput':

    print('Start inference given user input text...')

    from dataset.dataset_VQ_bodypart import whole2parts, parts2whole

    clip_text = [test_args.input_text]

    text = clip.tokenize(clip_text, truncate=True).cuda()
    feat_clip_text = clip_model.encode_text(text).float()

    index_motion = ParCo_transformer.sample(feat_clip_text[0:1], True)

    parts_pred_pose = ParCo_vqvae.forward_decoder(index_motion)
    pred_pose = parts2whole(parts_pred_pose)

    mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
    std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()

    pred_xyz = recover_from_ric((pred_pose*std + mean).float(), 22)
    xyz = pred_xyz.reshape(1, -1, 22, 3)

    npy_save_dir = os.path.join(infer_out_dir, 'motion.npy')
    gif_save_dir = os.path.join(infer_out_dir, 'skeleton_viz.gif')
    np.save(npy_save_dir, xyz.detach().cpu().numpy())
    pose_vis = plot_3d.draw_to_batch(xyz.detach().cpu().numpy(),clip_text, [gif_save_dir])
    logger.info('Inference Done!')


else:  # 'testset'

    print('Start inference on text dataset ...')

    # Prepare test set
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    val_loader = dataset_TM_eval_bodypart.DATALoader(args.dataname, True, 32, w_vectorizer)
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    eval_bodypart.evaluation_transformer_test_batch(
        infer_out_dir, val_loader, ParCo_vqvae, ParCo_transformer,
        logger, writer, 0, best_fid=1000, best_iter=0, best_div=100,
        best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_multi=0,
        clip_model=clip_model, eval_wrapper=eval_wrapper,
        draw=False, savegif=False, save=False, savenpy=True,
        mmod_gen_times=1, skip_mmod=True)

    logger.info('Inference Done!')