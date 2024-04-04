import argparse


vqvae_bodypart_cfg = {
    'default': dict(
        parts_code_nb={  # number of codes
            'Root': 512,
            'R_Leg': 512,
            'L_Leg': 512,
            'Backbone': 512,
            'R_Arm': 512,
            'L_Arm': 512,
        },
        parts_code_dim={  # Remember code_dim should be same to output_dim
            'Root': 64,
            'R_Leg': 128,
            'L_Leg': 128,
            'Backbone': 128,
            'R_Arm': 128,
            'L_Arm': 128,
        },
        parts_output_dim={  # dimension of encoder's output
            'Root': 64,
            'R_Leg': 128,
            'L_Leg': 128,
            'Backbone': 128,
            'R_Arm': 128,
            'L_Arm': 128,
        },
        parts_hidden_dim={  # hidden dimension of conv1d in encoder/decoder
            'Root': 64,
            'R_Leg': 128,
            'L_Leg': 128,
            'Backbone': 128,
            'R_Arm': 128,
            'L_Arm': 128,
        }

    ),

}

def get_args_parser(args=None):
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader  
    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')

    ## optimization
    parser.add_argument('--total-iter', default=200000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000, 400000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss-vel', type=float, default=0.1, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons-loss', type=str, default='l2', help='reconstruction loss')


    parser.add_argument("--vqvae-cfg", type=str, help="Base config for vqvae")

    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")

    # It is the number of downsampling block in the net, not the downsampling rate referred in the paper.
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output/', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')
    
    if args is None:
        return parser.parse_args()

    else:
        return parser.parse_args(args=args)


def get_vavae_test_args_parser():
    parser = argparse.ArgumentParser(description='Evaluate the body VQVAE_bodypart',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--vqvae-train-dir', type=str, help='VQVAE training directory')
    parser.add_argument('--select-vqvae-ckpt', type=str, help='Select which ckpt for use: [last, fid, div, top1, matching]')


    return parser.parse_args()



