import numpy as np
import pandas as pd



def ohlc_distance(x_df: pd.DataFrame, y: np.ndarray):
    df = x_df.copy()

    df['h_diss'] = abs(df.high - y)# / df.spread
    df['l_diss'] = abs(df.low - y) #/ df.spread
    df['diss'] = df[['h_diss', 'l_diss']].min(axis=1)
    diss = df['diss'].to_numpy()
    #diss[diss < 1] = 0
    return np.mean(diss)# , diss



def normalizer(time_series: np.ndarray):
    n_ts = time_series.copy()
    max_min = (np.max(time_series) - np.min(time_series))
    n_ts -= np.min(time_series)
    n_ts /= max_min
    return n_ts


def inv_normal(normal_time_series: np.ndarray, min_value, max_value):
    n_ts = normal_time_series.copy()
    max_min = (max_value - min_value)
    n_ts *= max_min
    n_ts += min_value
    return n_ts


def inv_diff(diff_time_series: np.ndarray, initial_value):
    reconstruct_series = np.zeros(len(diff_time_series) + 1)
    reconstruct_series[0] = initial_value
    for i in range(len(diff_time_series)):
        reconstruct_series[i + 1] = reconstruct_series[i] + diff_time_series[i]
    return reconstruct_series

##
def split_train_valid(df: pd.DataFrame, split_ratio: float = 0.1):
    split_ratio = np.clip(split_ratio, a_min=0.01, a_max=0.99)
    index_split = int(len(df) * (1 - split_ratio))
    return pd.DataFrame(df.iloc[:index_split, ]), pd.DataFrame(df.iloc[index_split - 1:, ])
##

def time_series_embedding(data, delay=1, dimension=2):
    "This function returns the Takens embedding of data with delay into dimension, delay*dimension must be < len(data)"
    if delay * dimension > len(data):
        raise NameError('Delay times dimension exceed length of data!')
    emd_ts = np.array([data[0:len(data) - delay * dimension]])
    for i in range(1, dimension):
        emd_ts = np.append(emd_ts, [data[i * delay:len(data) - delay * (dimension - i)]], axis=0)
    return emd_ts

# # TEST
# from LiveRate import ForexMarket
# import MetaTrader5 as mt5
# import matplotlib.pyplot as plt
#
# currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF']  # , 'CAD', 'AUD', ]  # 'NZD']
# fx = ForexMarket(currencies)
# xt = fx.get_all_rates(time_shift=60 * 3, time_frame=mt5.TIMEFRAME_M1, start_from=0)
#
# x = xt['EURUSD_i'].close
# # print(x)
# # print(split_train_valid(x,0.1))
# #
# # x1 = xt['EURUSD_i'].close.to_numpy()
# # x2 = xt['EURUSD_i'].high.to_numpy()
# # print(euclidean(x1,x2))
# # print(loss_dynamic_time_warping(x1,x2))
# # #
# diff_x = np.diff(x)
# n_x = normalizer(diff_x)
#
# inv_n_x = inv_normal(n_x, np.min(diff_x), np.max(diff_x))
# inv_diff_x = inv_diff(inv_n_x, x[0])
# plt.plot(n_x,'.--')
#
#
# #plt.plot(x.to_numpy())
#
# plt.show()
