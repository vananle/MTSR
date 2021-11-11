import argparse

ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']


def get_args():
    # create argument parser
    parser = argparse.ArgumentParser()

    # parameter for dataset
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='abilene_tm',
                        choices=['abilene_tm', 'geant_tm', 'brain_tm', 'renater_tm', 'surfnet_tm', 'uninett_tm'],
                        help='Dataset, (default abilene_tm)')
    parser.add_argument('--datapath', type=str, default='../../data_mssr/')
    parser.add_argument('--type', type=str, default='p2', choices=['p1', 'p2', 'p3'])

    parser.add_argument('--use_scaler', action='store_true')

    # Model
    parser.add_argument('--model', type=str, default='arima')
    # Wavenet
    parser.add_argument('--seq_len_x', type=int, default=216, help='')
    parser.add_argument('--dl_seq_len_x', type=int, default=12, help='')
    parser.add_argument('--seq_len_y', type=int, default=12, help='')

    parser.add_argument('--blocks', type=int, default=5, help='')
    parser.add_argument('--layers', type=int, default=2, help='')

    # loss
    parser.add_argument('--loss_fn', type=str, default='mae', choices=['mse', 'mae', 'mse_u', 'mae_u'])
    parser.add_argument('--lamda', type=float, default=2.0)

    # training
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--plot', action='store_true')

    # parameter for test_routing
    parser.add_argument('--ncf', default=10, type=int, help='default 10')
    parser.add_argument('--run_te', type=str, choices=['None', 'gwn_ls2sr', 'gt_ls2sr', 'p0', 'p1', 'p2', 'gwn_p2',
                                                       'p3', 'onestep', 'prophet', 'laststep', 'laststep_ls2sr',
                                                       'firststep', 'or', 'gwn_srls', 'gt_srls', 'srls_p0',
                                                       'gwn_cfr_topk'],
                        default='None')

    parser.add_argument('--timeout', type=float, default=1.0)
    parser.add_argument('--te_step', type=int, default=0)
    parser.add_argument('--nrun', type=int, default=3)

    # get args
    args = parser.parse_args()

    if args.type == 'p1':
        args.out_seq_len = args.seq_len_y
    elif args.type == 'p2':
        args.out_seq_len = 1
    elif args.type == 'p3':
        if args.seq_len_y % args.trunk != 0:
            args.seq_len_y = int(args.seq_len_y / args.trunk) * args.trunk
        args.out_seq_len = args.trunk

    return args


def print_args(args):
    print('-------------------------------------')
    print('[+] Time-series recovering experiment')
    if args.test:
        print('|--- Run Test')
    else:
        print('|--- Run Train')

    print('---------------------------------------------------------')
    print('[+] Time-series prediction experiment')
    print('---------------------------------------------------------')
    print('    - dataset                :', args.dataset)
    print('    - num_series             :', args.nSeries)
    print('    - test size              : {}x{}'.format(args.test_size, args.nSeries))
    print('    - log path               :', args.log_dir)
    print('---------------------------------------------------------')
    print('    - model                  :', args.model)
    print('----------------------------')
    print('    - type                   :', args.type)
    print('    - seq_len_x              :', args.seq_len_x)
    print('    - seq_len_y              :', args.seq_len_y)
    print('    - out_seq_len            :', args.out_seq_len)
    print('---------------------------------------------------------')
    print('    - test_batch_size        :', args.test_batch_size)
    print('    - plot_results           :', args.plot)
    print('---------------------------------------------------------')
    print('    - run te                 :', args.run_te)
    print('    - te_step                :', args.te_step)
    print('---------------------------------------------------------')
