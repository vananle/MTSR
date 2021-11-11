import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt
import os
import numpy as np


def plot_results(plot_dir, x, y, yhat, plot_id):
    if plot_id >= 5:
        return
    print(x.shape, y.shape, yhat.shape)
    gt = np.concatenate([x, y], axis=1)
    preds = np.concatenate([x, yhat], axis=1)
    for n in range(x.shape[-1]):
        _plot_dir = os.path.join(plot_dir, 'flow_{}/'.format(n))
        if not os.path.exists(_plot_dir):
            os.makedirs(_plot_dir)
        for i in range(1, x.shape[1]):
            f, a = plt.subplots(figsize=(11, 9))
            plt.plot(gt[i, :, n], color='black', label='X')
            plt.plot(preds[i, :, n], color='red', label='Xhat')
            plt.legend()
            plt_name = os.path.join(_plot_dir, '{}.eps'.format(plot_id * x.shape[0] + i))
            plt.savefig(plt_name, dpi=300)
            plt.close()
