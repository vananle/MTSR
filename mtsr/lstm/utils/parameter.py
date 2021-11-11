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
    parser.add_argument('--type', type=str, default='p2', choices=['p2'])

    parser.add_argument('--tod', action='store_true')
    parser.add_argument('--ma', action='store_true')
    parser.add_argument('--mx', action='store_true')

    # Model
    # Graph
    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--seq_len_x', type=int, default=36, choices=[6, 12, 18, 24, 30, 36, 48],
                        help='input length default 64')
    parser.add_argument('--seq_len_y', type=int, default=36, choices=[6, 12, 18, 24, 30, 36, 48],
                        help='routing cycle 12')

    parser.add_argument('--layers', type=int, default=2, help='')
    parser.add_argument('--in_dim', type=int, default=1, help='')
    parser.add_argument('--out_dim', type=int, default=1, help='')
    parser.add_argument('--hidden', type=int, default=64, help='Number of channels for internal conv')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')

    # loss
    parser.add_argument('--loss_fn', type=str, default='mae', choices=['mse', 'mae', 'mse_u', 'mae_u'])
    parser.add_argument('--lamda', type=float, default=2.0)

    # training
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--epochs', type=int, default=300, help='')
    parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='learning rate')
    parser.add_argument('--patience', type=int, default=20, help='quit if no improvement after this many iterations')
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

    args.seq_len_y = args.seq_len_x

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
    print('    - log path               :', args.log_dir)
    print('---------------------------------------------------------')
    print('    - model                  :', args.model)
    print('    - layers                 :', args.layers)
    print('    - hidden                 :', args.hidden)
    print('----------------------------')
    print('    - type                   :', args.type)
    print('    - seq_len_x              :', args.seq_len_x)
    print('    - seq_len_y              :', args.seq_len_y)
    print('    - out_seq_len            :', args.out_seq_len)
    print('---------------------------------------------------------')
    print('    - device                 :', args.device)
    print('    - train_batch_size       :', args.train_batch_size)
    print('    - val_batch_size         :', args.val_batch_size)
    print('    - test_batch_size        :', args.test_batch_size)
    print('    - epochs                 :', args.epochs)
    print('    - learning_rate          :', args.learning_rate)
    print('    - patience               :', args.patience)
    print('    - plot_results           :', args.plot)
    print('---------------------------------------------------------')
    print('    - run te                 :', args.run_te)
    print('    - te_step                :', args.te_step)
    print('    - ncf                    :', args.ncf)
    print('---------------------------------------------------------')
