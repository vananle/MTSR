import warnings

import numpy as np
from pmdarima.arima import AutoARIMA

warnings.filterwarnings('ignore')


class AutoArima:
    def __init__(self, args):
        self.model = AutoARIMA()
        self.seq_len_x = args.seq_len_x
        self.out_seq_len = args.out_seq_len
        self.args = args

    def predict(self, x):
        # input [batch, in_seq_len, n]

        b, seq_x, n = x.shape
        x = np.reshape(x, [-1, seq_x])

        n_samples, _ = x.shape

        xhat = []
        for i in range(n_samples):
            y = self.model.fit_predict(x[i], n_periods=self.out_seq_len)
            xhat.append(y)

        xhat = np.stack(xhat, axis=-1)
        xhat = np.reshape(xhat, (b, self.out_seq_len, n))

        return xhat  # (b, out_len, n)
