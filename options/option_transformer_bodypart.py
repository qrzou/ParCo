import argparse

trans_bodypart_cfg = {
    'default': dict(
        # Common transformer config
        parts_code_nb={  # size of part codebooks
            'Root': 512,
            'R_Leg': 512,
            'L_Leg': 512,
            'Backbone': 512,
            'R_Arm': 512,
            'L_Arm': 512,
        },
        parts_embed_dim={  # dimension (size) of transformer attention block
            'Root': 256,
            'R_Leg': 256,
            'L_Leg': 256,
            'Backbone': 256,
            'R_Arm': 256,
            'L_Arm': 256,
        },
        # Fuse V1 config
        num_mlp_layers=3,
        # FuseV2 config
        fusev2_sub_mlp_out_features={
            'Root': 64,
            'R_Leg': 64,
            'L_Leg': 64,
            'Backbone': 64,
            'R_Arm': 64,
            'L_Arm': 64,
        },
        fusev2_sub_mlp_num_layers=2,
        fusev2_head_mlp_num_layers=2,
    ),
    'small': dict(
        # Common transformer config
        parts_code_nb={  # size of part codebooks
            'Root': 512,
            'R_Leg': 512,
            'L_Leg': 512,
            'Backbone': 512,
            'R_Arm': 512,
            'L_Arm': 512,
        },
        parts_embed_dim={  # dimension (size) of transformer attention block
            'Root': 128,
            'R_Leg': 128,
            'L_Leg': 128,
            'Backbone': 128,
            'R_Arm': 128,
            'L_Arm': 128,
        },
        # Fuse V1 config
        num_mlp_layers=3,
        # FuseV2 config
        fusev2_sub_mlp_out_features={
            'Root': 64,
            'R_Leg': 64,
            'L_Leg': 64,
            'Backbone': 64,
            'R_Arm': 64,
            'L_Arm': 64,
        },
        fusev2_sub_mlp_num_layers=2,
        fusev2_head_mlp_num_layers=2,
    ),
    'tiny': dict(
        # Common transformer config
        parts_code_nb={  # size of part codebooks
            'Root': 512,
            'R_Leg': 512,
            'L_Leg': 512,
            'Backbone': 512,
            'R_Arm': 512,
            'L_Arm': 512,
        },
        parts_embed_dim={  # dimension (size) of transformer attention block
            'Root': 64,
            'R_Leg': 64,
            'L_Leg': 64,
            'Backbone': 64,
            'R_Arm': 64,
            'L_Arm': 64,
        },
        # Fuse V1 config
        num_mlp_layers=3,
        # FuseV2 config
        fusev2_sub_mlp_out_features={
            'Root': 64,
            'R_Leg': 64,
            'L_Leg': 64,
            'Backbone': 64,
            'R_Arm': 64,
            'L_Arm': 64,
        },
        fusev2_sub_mlp_num_layers=2,
        fusev2_head_mlp_num_layers=2,
    ),
}

def get_args_parser(args=None):
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## dataloader
    
    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    parser.add_argument('--seq-len', type=int, default=64, help='training motion length')
    
    ## optimization
    parser.add_argument('--total-iter', default=100000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[60000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--decay-option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding (code book size)")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")

    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=3, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')

    ## gpt arch
    parser.add_argument("--block-size", type=int, default=25, help="seq len")
    parser.add_argument("--embed-dim-gpt", type=int, default=512, help="embedding dimension")
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-layers", type=int, default=2, help="nb of transformer layers")
    parser.add_argument("--n-head-gpt", type=int, default=8, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")

    # [Part Coordinating]
    parser.add_argument("--trans-cfg", type=str, help="Base config for our part coordinating transformer")

    parser.add_argument("--sync-part-maskaug", action='store_true', help="all parts use the same masking augmentation")

    parser.add_argument("--no-fuse", action='store_true', help="stop using Part-Coordinating module")
    parser.add_argument("--fuse-ver", type=str, default='V1_2', choices=['V1', 'V1_2', 'V1_3', 'V1_4', 'V2', 'V2_2'], help="Fuse module structure")
    parser.add_argument("--alpha", type=float, default=1.0, help="Part Coordinating strength.")


    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume vq pth')
    parser.add_argument("--resume-trans", type=str, default=None, help='resume gpt pth')
    
    
    ## output directory

    # parser.add_argument('--out-dir', type=str, default='output/', help='output directory')
    parser.add_argument('--select-vqvae-ckpt', type=str, help='Select which ckpt for use: [last, fid, div, top1, matching]')
    parser.add_argument('--vqvae-train-dir', type=str, help='use which vqvae for train Transformer')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    parser.add_argument("--use-existing-vq-data", action='store_true', help="Use existing quantized motion data.")
    parser.add_argument("--existing-vq-data-dir", type=str, default=None, help='Directory of existing quantized motion data')

    # parser.add_argument('--vq-name', type=str, default='exp_debug', help='name of the generated dataset .npy, will create a file inside out-dir')

    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=5000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    parser.add_argument('--pkeep', type=float, default=1.0, help='keep rate for gpt training, lower pkeep, higher mask rate')
    parser.add_argument("--use-pkeep-scheduler", action='store_true', help="Use pkeep scheduler")

    if args is None:
        return parser.parse_args()

    else:
        return parser.parse_args(args=args)
