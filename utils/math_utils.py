import numpy as np
import tensorflow as tf 

def uniform(shape, scale=1.0, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=0.0, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def z_score(x, mean, std):
    return (x - mean) / std

def z_inverse(x, mean, std):
    return x * std + mean

def MAPE(v, v_prediction):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    return np.mean(np.abs(v_prediction - v) / (v + 1e-5))

def RMSE(v, v_prediction):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_prediction - v) ** 2))

def MAE(v, v_prediction):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_prediction - v))

def evaluation(y, y_prediction, x_stats):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    dim = len(y_prediction.shape)

    if dim == 3:
        # single_step case
        v = z_inverse(y, x_stats['mean'], x_stats['std'])
        v_prediction = z_inverse(y_prediction, x_stats['mean'], x_stats['std'])
        return np.array([MAPE(v, v_prediction), MAE(v, v_prediction), RMSE(v, v_prediction)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [time_step, batch_size, n_route, 1]
        y = np.swapaxes(y, 0, 1)
        # recursively call
        for i in range(y_prediction.shape[0]):
            tmp_res = evaluation(y[i], y_prediction[i], x_stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)