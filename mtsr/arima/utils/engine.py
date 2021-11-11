import pandas as pd
from tqdm import tqdm

from .metric import *


class Trainer():
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    @classmethod
    def from_args(cls, model, scaler, args):
        return cls(model, scaler)

    # def train(self, input, real_val):
    #     self.model.train()
    #     self.optimizer.zero_grad()
    #     # input = torch.nn.functional.pad(input, (1, 0, 0, 0))
    #
    #     output = self.model(input)  # now, output = [bs, seq_y, n]
    #     predict = self.scaler.inverse_transform(output)
    #
    #     loss = self.lossfn(predict, real_val)
    #     rse, mae, mse, mape, rmse = calc_metrics(predict, real_val)
    #     loss.backward()
    #
    #     if self.clip is not None:
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
    #     self.optimizer.step()
    #     return loss.item(), rse.item(), mae.item(), mse.item(), mape.item(), rmse.item()
    #
    # def _eval(self, input, real_val):
    #     self.model.eval()
    #
    #     output = self.model(input)  # now, output = [bs, seq_y, n]
    #
    #     predict = self.scaler.inverse_transform(output)
    #
    #     predict = torch.clamp(predict, min=0., max=10e10)
    #     loss = self.lossfn(predict, real_val)
    #     rse, mae, mse, mape, rmse = calc_metrics(predict, real_val)
    #
    #     return loss.item(), rse.item(), mae.item(), mse.item(), mape.item(), rmse.item()

    def test(self, test_loader, model, out_seq_len):
        outputs = []
        y_real = []
        x_gt = []
        y_gt = []
        for i, batch in tqdm(enumerate(test_loader)):
            x = batch['x']  # [b, seq_x, n]
            y = batch['y']  # [b, seq_y, n]

            preds = model.predict(x)
            if self.scaler is not None:
                preds = self.scaler.inverse_transform(preds)  # [bs, out_seq_len, n]
            outputs.append(preds)
            y_real.append(y)
            x_gt.append(batch['x_gt'])
            y_gt.append(batch['y_gt'])

        yhat = np.concatenate(outputs, axis=0)
        y_real = np.concatenate(y_real, axis=0)
        x_gt = np.concatenate(x_gt, axis=0)
        y_gt = np.concatenate(y_gt, axis=0)
        test_met = []

        yhat[yhat < 0.0] = 0.0

        for i in range(out_seq_len):
            pred = yhat[:, i, :]
            pred[pred < 0.0] = 0.0
            real = y_real[:, i, :]
            test_met.append([x for x in calc_metrics_np(pred, real)])
        test_met_df = pd.DataFrame(test_met, columns=['rse', 'mae', 'mse', 'mape', 'rmse']).rename_axis('t')
        return test_met_df, x_gt, y_gt, y_real, yhat

    # def eval(self, val_loader):
    #     """Run validation."""
    #     val_loss, val_rse, val_mae, val_mse, val_mape, val_rmse = [], [], [], [], [], []
    #     for _, batch in enumerate(val_loader):
    #         x = batch['x']  # [b, seq_x, n, f]
    #         y = batch['y']  # [b, seq_y, n]
    #
    #         metrics = self._eval(x, y)
    #         val_loss.append(metrics[0])
    #         val_rse.append(metrics[1])
    #         val_mae.append(metrics[2])
    #         val_mse.append(metrics[3])
    #         val_mape.append(metrics[4])
    #         val_rmse.append(metrics[5])
    #
    #     return val_loss, val_rse, val_mae, val_mse, val_mape, val_rmse
