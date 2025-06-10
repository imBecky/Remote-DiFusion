import argparse


def get_argument_parse():
    parse = argparse.ArgumentParser(description='parameters')
    parse.add_argument('--trial_run', type=int, default=2, help='Experiment number of the trial run.')
    parse.add_argument('--dataset', type=str, default='SZU_R1', help='DFC Houston 2018 | SZUTreeData')
    parse.add_argument('--lr1', type=float, default=0.0005, help='learning rate for noise predictor')
    parse.add_argument('--lr2', type=float, default=0.0005, help='learning rate for classifier')
    parse.add_argument('--lr3', type=float, default=0.0001, help='learning rate for GAN block')
    parse.add_argument('--bs', type=int, default=2, help='batch_size')
    parse.add_argument('--seed', type=int, default=13, help='default seed = 13')
    parse.add_argument('--T', type=int, default=1000, help='time steps for diffusion procedure')
    parse.add_argument('--if_small_dataset', type=int, default=1)
    parse.add_argument('--epoch', type=int, default=30)
    parse.add_argument('--dataset_ratio', type=float, default=0.8, help='Split ratio of train and test dataset')
    parse.add_argument('--image_size', type=int, default=32, help="")  # TODO: complete the help
    parse.add_argument('--feature_channels', type=int, default=1, help="")  # TODO: complete the help
    parse.add_argument('--dim_mults', type=tuple, default=(1, 2, 4), help='dims of ?')  # TODO: complete the help
    parse.add_argument('--dr', type=float, default=0.5, help='dropout rate of classifier')
    parse.add_argument('--betas', type=str, default='0.5,0.999',
                       help='Adam optimizer betas parameters (beta1,beta2). Default: 0.5,0.999')
    parse.add_argument('--log_dir', default='./logs')
    args = parse.parse_args()
    # 转换 betas 字符串为浮点数元组
    try:
        args.betas = tuple(map(float, args.betas.split(',')))
        if len(args.betas) != 2:
            raise ValueError
    except:
        raise argparse.ArgumentTypeError(
            "betas must be two comma-separated floats (e.g. '0.5,0.999')")
    return args


args = get_argument_parse()
