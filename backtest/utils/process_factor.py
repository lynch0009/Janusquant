import numpy as np
'''
因子数据处理： 去极值，标准化
'''


def three_sigma(factor):
    mean = factor.mean()
    std = factor.std()
    up = mean+3*std
    down = mean-3*std
    factor = np.where(factor > up, up, factor)
    factor = np.where(factor < down, down, factor)
    return factor


def stand(factor):
    mean = factor.mean()
    std = factor.std()
    return (factor-mean)/std

