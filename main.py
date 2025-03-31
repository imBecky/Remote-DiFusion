import argparse
from dataset import data_report


def main(args):
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    if args.trial_run == 0:
        data_report('./dataset/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='实验参数配置')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    args = parser.parse_args()
    main(args)
