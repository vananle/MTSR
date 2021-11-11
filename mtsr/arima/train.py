import sys
import time

sys.path.append('..')
from routing import *

import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
import models
import numpy as np
import torch
import utils


def main(args, **model_kwargs):
    device = torch.device(args.device)
    args.device = device
    if 'abilene' in args.dataset:
        args.nNodes = 12
        args.day_size = 288
    elif 'geant' in args.dataset:
        args.nNodes = 22
        args.day_size = 96
    elif 'brain' in args.dataset:
        args.nNodes = 9
        args.day_size = 1440
    elif 'sinet' in args.dataset:
        args.nNodes = 74
        args.day_size = 288
    elif 'renater' in args.dataset:
        args.nNodes = 30
        args.day_size = 288
    elif 'surfnet' in args.dataset:
        args.nNodes = 50
        args.day_size = 288
    elif 'uninett' in args.dataset:
        args.nNodes = 74
        args.day_size = 288
    else:
        raise ValueError('Dataset not found!')

    test_loader = utils.get_dataloader(args)

    args.test_size, args.nSeries = test_loader.dataset.gt_data_set.shape

    in_dim = 1
    args.in_dim = in_dim

    model = models.get_model(args)
    logger = utils.Logger(args)

    engine = utils.Trainer.from_args(model, test_loader.dataset.scaler, args)

    utils.print_args(args)

    if not args.test:
        test_met_df, x_gt, y_gt, y_real, yhat = engine.test(test_loader, engine.model, args.out_seq_len)
        test_met_df.round(6).to_csv(os.path.join(logger.log_dir, 'test_metrics.csv'))
        print('Prediction Accuracy:')
        print(utils.summary(logger.log_dir))
        np.save(os.path.join(logger.log_dir, 'x_gt'), x_gt)
        np.save(os.path.join(logger.log_dir, 'y_gt'), y_gt)
        np.save(os.path.join(logger.log_dir, 'y_real'), y_real)
        np.save(os.path.join(logger.log_dir, 'yhat'), yhat)

    else:
        x_gt = np.load(os.path.join(logger.log_dir, 'x_gt.npy'))
        y_gt = np.load(os.path.join(logger.log_dir, 'y_gt.npy'))
        y_real = np.load(os.path.join(logger.log_dir, 'y_real.npy'))
        yhat = np.load(os.path.join(logger.log_dir, 'yhat.npy'))

    if args.plot:
        logger.plot(x_gt, y_real, yhat)

    # run TE
    if args.run_te:
        run_te(x_gt, y_gt, yhat, args)


if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
