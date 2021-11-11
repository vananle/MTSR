from .arima import AutoArima


def get_model(args):
    model = AutoArima(args)
    return model
