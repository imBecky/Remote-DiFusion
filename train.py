import argparse
from dataset import data_report


def main(args):
    if args.trial_run == 0:
        data_report('./dataset/')


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='parameters')
    parse.add_argument('--trial_run', type=int, default=0, help='Experiment number of the trial run.')
    parse.add_argument('--dataset', type=str, default='Houston', help='DFC Houston 2018 | SZUTreeData')
    parse.add_argument('--lr1', type=float, default=0.0005, help='learning rate for noise predictor')
    parse.add_argument('--lr2', type=float, default=0.0005, help='learning rate for discriminator')
    parse.add_argument('--lr3', type=float, default=0.005, help='learning rate for classifier')
    parse.add_argument('--bs', type=int, default=30, help='batch_size')
    parse.add_argument('--seed', type=int, default=13, help='default seed = 13')
    parse.add_argument('--T', type=int, default=1000, help='time steps for diffusion procedure')
    parse.add_argument('--image_size', type=int, default=32)
    parse.add_argument('--if_small_dataset', type=int, default=1)
    parse.add_argument('--dim_mults', type=tuple, default=(1, 2, 4), help='dims of ?')  # TODO: complete the help
    parse.add_argument('--epoch', type=int, default=30)
    parse.add_argument('--log_dir', default='./logs')
    args = parse.parse_args()
    main(args)
