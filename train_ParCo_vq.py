import os
import json
import re
# import warnings

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import dataset_VQ_bodypart, dataset_TM_eval_bodypart
from models import vqvae_bodypart as vqvae
from models.evaluator_wrapper import EvaluatorModelWrapper

from options.get_eval_option import get_opt
import options.option_vq_bodypart as option_vq
from options.option_vq_bodypart import vqvae_bodypart_cfg

import utils.losses as losses
import utils.utils_model as utils_model
import utils.eval_bodypart as eval_bodypart
from utils.word_vectorizer import WordVectorizer

# warnings.filterwarnings('ignore')


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


##### ---- Parse args ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
args.vqvae_arch_cfg = vqvae_bodypart_cfg[args.vqvae_cfg]

##### ---- Exp dirs ---- #####
"""
Directory of our exp:
./output  (arg.out_dir)
 ├── 00000-DATASET  (exp_number + dataset_name)
 │   └── VQVAE-EXP_NAME-DESC  (VQVAE + args.exp_name + desc)
 │       ├── events.out.XXX
 │       ├── net_best_XXX.pth
 │       ...
 │       ├── run.log
 │       ├── test_vqvae
 │       │   ├── ...
 │       │   ...
 │       ├── 0000-Trans-EXP_NAME-DESC  (stage2_exp_number + Trans + args.exp_name + desc)
 │       │   ├── quantized_dataset  (The quantized motion using VQVAE)
 │       │   ├── events.out.XXX
 │       │   ├── net_best_XXX.pth
 │       │   ...
 │       │   ├── run.log
 │       │   └── test_trans
 │       │       ├── ...
 │       │       ...
 │       ├── 0001-Trans-EXP_NAME-DESC
 │       ...
 ├── 00001-DATASET  (exp_number + dataset_name)
 ...
"""
# [Prepare description]
desc = args.dataname  # dataset
desc += f'-{args.vqvae_cfg}'

# Pick output directory.
prev_run_dirs = []

outdir = args.out_dir
if os.path.isdir(outdir):
    prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
cur_run_id = max(prev_run_ids, default=-1) + 1
args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{args.dataname}-{args.exp_name}', f'VQVAE-{args.exp_name}-{desc}')
assert not os.path.exists(args.run_dir)
print('Creating output directory...')
os.makedirs(args.run_dir)


##### ---- Logger ---- #####
logger = utils_model.get_logger(args.run_dir)
writer = SummaryWriter(args.run_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

# save the training config
args.args_save_dir = os.path.join(args.run_dir, 'train_config.json')
args_dict = vars(args)
with open(args.args_save_dir, 'wt') as f:
    json.dump(args_dict, f, indent=4)


w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit':
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
    args.nb_joints = 21
    
else :
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
print('\n\n===> Constructing dataset and dataloader...\n\n')
train_loader = dataset_VQ_bodypart.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t)

train_loader_iter = dataset_VQ_bodypart.cycle(train_loader)

val_loader = dataset_TM_eval_bodypart.DATALoader(args.dataname, False,
                                        32,
                                        w_vectorizer,
                                        unit_length=2**args.down_t)

##### ---- Network ---- #####
print('\n\n===> Constructing network...')
net = vqvae.HumanVQVAEBodyPart(
    args,  # use args to define different parameters in different quantizers
    args.vqvae_arch_cfg['parts_code_nb'],
    args.vqvae_arch_cfg['parts_code_dim'],
    args.vqvae_arch_cfg['parts_output_dim'],
    args.vqvae_arch_cfg['parts_hidden_dim'],
    args.down_t,
    args.stride_t,
    args.depth,
    args.dilation_growth_rate,
    args.vq_act,
    args.vq_norm
)


if args.resume_pth:
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.train()
net.cuda()

##### ---- Optimizer & Scheduler ---- #####
print('\n===> Constructing optimizer, scheduler, and Loss...')
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
Loss = losses.ReConsLossBodyPart(args.recons_loss, args.nb_joints)

##### ------ warm-up ------- #####
print('\n===> Start warm-up training\n\n')

avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

for nb_iter in range(1, args.warm_up_iter):
    
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

    gt_parts = next(train_loader_iter)
    for i in range(len(gt_parts)):
        gt_parts[i] = gt_parts[i].cuda().float()

    pred_parts, loss_commit_list, perplexity_list = net(gt_parts)

    pred_parts_vel = dataset_VQ_bodypart.get_each_part_vel(
        pred_parts, mode=args.dataname)
    gt_parts_vel = dataset_VQ_bodypart.get_each_part_vel(
        gt_parts, mode=args.dataname)

    loss_motion_list = Loss(pred_parts, gt_parts)  # parts motion reconstruction loss
    loss_vel_list = Loss.forward_vel(pred_parts_vel, gt_parts_vel)  # parts velocity recon loss

    loss_motion = losses.gather_loss_list(loss_motion_list)
    loss_commit = losses.gather_loss_list(loss_commit_list)
    loss_vel = losses.gather_loss_list(loss_vel_list)

    loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_motion.item()
    perplexity = losses.gather_loss_list(perplexity_list)
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter == 0:
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
    eval_bodypart.evaluation_vqvae(
        args.run_dir, val_loader, net, logger, writer, 0,
        best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100,
        eval_wrapper=eval_wrapper)

##### ---- Training ---- #####
print('\n\n===> Start training\n\n')
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

for nb_iter in range(1, args.total_iter + 1):

    gt_parts = next(train_loader_iter)
    for i in range(len(gt_parts)):
        gt_parts[i] = gt_parts[i].cuda().float()

    pred_parts, loss_commit_list, perplexity_list = net(gt_parts)

    pred_parts_vel = dataset_VQ_bodypart.get_each_part_vel(
        pred_parts, mode=args.dataname)
    gt_parts_vel = dataset_VQ_bodypart.get_each_part_vel(
        gt_parts, mode=args.dataname)

    loss_motion_list = Loss(pred_parts, gt_parts)  # parts motion reconstruction loss
    loss_vel_list = Loss.forward_vel(pred_parts_vel, gt_parts_vel)  # parts velocity recon loss


    loss_motion = losses.gather_loss_list(loss_motion_list)
    loss_commit = losses.gather_loss_list(loss_commit_list)
    loss_vel = losses.gather_loss_list(loss_vel_list)

    loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_motion.item()
    perplexity = losses.gather_loss_list(perplexity_list)
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter == 0:
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        writer.add_scalar('./Train/L1', avg_recons, nb_iter)
        writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
        
        logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,

    if nb_iter % args.eval_iter == 0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
            eval_bodypart.evaluation_vqvae(
                args.run_dir, val_loader, net, logger, writer, nb_iter,
                best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching,
                eval_wrapper=eval_wrapper)
        