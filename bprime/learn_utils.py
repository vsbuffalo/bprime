import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

MEAN_LOSS_FUNCS = {'mae': mean_absolute_error,
                   'mape': mean_absolute_percentage_error,
                   'mse': mean_squared_error}


def get_loss_func(loss):
    try:
        return LOSS_FUNCS[loss.lower()]
    except KeyError:
        opts = ', '.join(LOSS_FUNCS)
        raise KeyError(f"'{loss}' is not a loss function (options: {opts})")


def absolute_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return np.abs(y_true - y_pred)

def bias(y_true, y_pred):
    # B(T, θ) = E(T) - θ; this is averaged later
    assert y_true.shape == y_pred.shape
    return y_pred - y_true

def absolute_percentage_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return np.abs(y_true - y_pred) / y_pred

def squared_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return (y_true - y_pred)**2

LOSS_FUNCS = {'mae': absolute_error,
              'mape': absolute_percentage_error,
              'bias': bias,
              'mse': squared_error}


