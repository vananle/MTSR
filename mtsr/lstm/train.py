import sys

sys.path.append('..')

import time
import models
import torch
import utils
from tqdm import trange
from routing import *

import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


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

    train_loader, val_loader, test_loader, total_timesteps, total_series = utils.get_dataloader(args)

    args.test_size = test_loader.dataset.nsample
    args.te_step = args.test_size
    args.nSeries = total_series

    in_dim = 1
    if args.tod:
        in_dim += 1
    if args.ma:
        in_dim += 1
    if args.mx:
        in_dim += 1

    args.in_dim = in_dim

    model = models.get_model(args)
    logger = utils.Logger(args)

    engine = utils.Trainer.from_args(model, test_loader.dataset.scaler, args)

    utils.print_args(args)

    if not args.test:
        iterator = trange(args.epochs)

        try:
            if os.path.isfile(logger.best_model_save_path):
                print('Model checkpoint exist!')
                print('Load model checkpoint? (y/n)')
                _in = input()
                if _in == 'y' or _in == 'yes':
                    print('Loading model...')
                    engine.model.load_state_dict(torch.load(logger.best_model_save_path))
                else:
                    print('Training new model')

            for epoch in iterator:
                train_loss, train_rse, train_mae, train_mse, train_mape, train_rmse = [], [], [], [], [], []
                for iter, batch in enumerate(train_loader):

                    x = batch['x']  # [b, seq_x, n, f]
                    y = batch['y']  # [b, seq_y, n]

                    if y.max() == 0: continue
                    loss, rse, mae, mse, mape, rmse = engine.train(x, y)
                    train_loss.append(loss)
                    train_rse.append(rse)
                    train_mae.append(mae)
                    train_mse.append(mse)
                    train_mape.append(mape)
                    train_rmse.append(rmse)

                engine.scheduler.step()
                with torch.no_grad():
                    val_loss, val_rse, val_mae, val_mse, val_mape, val_rmse = engine.eval(val_loader)
                m = dict(train_loss=np.mean(train_loss), train_rse=np.mean(train_rse),
                         train_mae=np.mean(train_mae), train_mse=np.mean(train_mse),
                         train_mape=np.mean(train_mape), train_rmse=np.mean(train_rmse),
                         val_loss=np.mean(val_loss), val_rse=np.mean(val_rse),
                         val_mae=np.mean(val_mae), val_mse=np.mean(val_mse),
                         val_mape=np.mean(val_mape), val_rmse=np.mean(val_rmse))

                description = logger.summary(m, engine.model)

                if logger.stop:
                    break

                description = 'Epoch: {} '.format(epoch) + description
                iterator.set_description(description)
        except KeyboardInterrupt:
            pass
    else:
        # Metrics on test data
        engine.model.load_state_dict(torch.load(logger.best_model_save_path))
        with torch.no_grad():
            test_met_df, x_gt, y_gt, y_real, yhat = engine.test(test_loader, engine.model, args.out_seq_len)
            test_met_df.round(6).to_csv(os.path.join(logger.log_dir, 'summarized_test_metrics.csv'))
            print('Prediction Accuracy:')
            print(test_met_df)

            test_met = []
            for t in range(yhat.shape[0]):
                for i in range(yhat.shape[1]):
                    pred = yhat[t, i, :]
                    real = y_real[t, i, :]
                    test_met.append([x.item() for x in utils.calc_metrics(pred, real)])
            test_met_df = pd.DataFrame(test_met, columns=['rse', 'mae', 'mse', 'mape', 'rmse']).rename_axis('t')
            test_met_df.round(6).to_csv(os.path.join(logger.log_dir, 'test_metrics.csv'))

        x_gt = x_gt.cpu().data.numpy()  # [timestep, seq_x, seq_y]
        y_gt = y_gt.cpu().data.numpy()
        yhat = yhat.cpu().data.numpy()
        y_real = y_real.cpu().data.numpy()
        np.save(os.path.join(logger.log_dir, 'x_gt_test'), x_gt)
        np.save(os.path.join(logger.log_dir, 'y_gt_test'), y_gt)
        np.save(os.path.join(logger.log_dir, 'y_hat_test'), yhat)
        np.save(os.path.join(logger.log_dir, 'y_real_test'), y_real)

        # run TE
        if args.run_te != 'None':
            run_te(x_gt, y_gt, yhat, args)


from datetime import date

if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())
