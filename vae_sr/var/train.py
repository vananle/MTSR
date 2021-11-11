import os.path
import time
import warnings
from datetime import date

import tensorflow as tf
import torch

import models
import utils
from ..routing import *

# sys.path.append('..')

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def main(args, **model_kwargs):
    device = torch.device(args.device)
    args.device = device
    if 'abilene' in args.dataset:
        args.nNodes = 12
        args.nLinks = 30
        args.day_size = 288
        scale = 100.0
    elif 'geant' in args.dataset:
        args.nNodes = 22
        args.nLinks = 72
        args.day_size = 96
        scale = 100.0
    elif 'brain' in args.dataset:
        args.nNodes = 9
        args.nLinks = 28
        args.day_size = 1440
    else:
        raise ValueError('Dataset not found!')
    logger = utils.Logger(args)
    X_train, X_val, L_train, L_val, A_train, A_val = utils.load_traindata(args)

    X_train = X_train / args.scale
    X_val = X_val / args.scale

    model = models.VAE(args=args, X_val=X_val)

    model.compile(optimizer=tf.keras.optimizers.Adam())
    print(model.encoder.summary())
    print(model.decoder.summary())
    if not args.test:
        history = model.fit(X_train, epochs=args.epochs, batch_size=args.train_batch_size,
                            callbacks=[utils.EarlyStoppingAtMinLoss()], )
        model.encoder.save(os.path.join(args.log_dir, 'encoder'))
        model.decoder.save(os.path.join(args.log_dir, 'decoder'))

    test_traffic = utils.load_test_traffic(args)
    graph = load_network_topology(args.dataset, args.datapath)
    vae_ls2sr(test_traffic=test_traffic, vae=model, graph=graph, args=args)


if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())
