import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import datetime
import matplotlib as mpl
from matplotlib.font_manager import FontProperties


def plot_origin_result():
    ori_result = pd.read_csv(r'C:\Users\DengTao\Desktop\明理杯\result_origin.csv', index_col=0, encoding='gbk')
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x='MSOE of upstream traffic', y='MSOE of downstream traffic', data=ori_result, hue='Imputation Method',
                    style='Neural Network', style_order=['GRU', 'LSTM', 'RNN-tanh', 'RNN-ReLU', 'GRU-D'],  # 用标记点形状表示的变量
                    palette=sns.color_palette('Accent', 7),
                    )  # 修改图标色板
    currentAxis = plt.gca()
    rect = patches.Rectangle((0, 0.001), 0.002, 0.001, linewidth=1, edgecolor='r', facecolor='none')
    currentAxis.add_patch(rect)
    # plt.legend(loc='lower right')
    # plt.tight_layout()
    xticklabels = ['$' + str(np.around(j, 1)) + r'\times10^{-2}$' for j in np.arange(0, 5, 1)]
    yticklabels = ['$' + str(np.around(j, 1)) + r'\times10^{-3}$' for j in np.arange(1, 7, 1)]
    plt.xticks(ticks=np.arange(0.00, 5e-2, 1e-2), labels=xticklabels)
    plt.yticks(ticks=np.arange(1e-3, 7e-3, 1e-3), labels=yticklabels)

    plt.savefig(r'C:\Users\DengTao\Desktop\明理杯\ori_result.jpg', dpi=150, bbox_inches='tight')


def plot_miner_result():
    result = pd.read_csv(r'C:\Users\DengTao\Desktop\明理杯\result.csv', index_col=0, encoding='gbk')
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x='MSOE of upstream traffic', y='MSOE of downstream traffic', data=result, hue='Imputation Method',
                    style='Neural Network', s=90, style_order=['GRU', 'LSTM', 'RNN-tanh', 'RNN-ReLU', 'GRU-D'],  # 用标记点形状表示的变量
                    palette=sns.color_palette('Accent', 7),
                    )  # 修改图标色板
    yticklabels = ['$1.' + str(i) + r'\times10^{-3}$' for i in range(2, 10)]
    xticklabels = ['$' + str(np.around(j, 1)) + r'\times10^{-3}$' for j in np.arange(0.8, 1.6, 0.1)]
    plt.xticks(ticks=np.arange(0.0008, 0.0016, 1e-4), labels=xticklabels)
    plt.yticks(ticks=np.arange(0.0012, 0.0019, 1e-4), labels=yticklabels)
    # plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.01))
    # plt.legend(loc='lower right')
    # plt.tight_layout()
    plt.savefig(r'C:\Users\DengTao\Desktop\明理杯\result.jpg', dpi=150, bbox_inches='tight')


def plot_missing_rate():
    mask = np.load(r'F:\git_repo\deepTimeSeries\data\mask_20.npy')
    time = np.sort(np.load(r'F:\git_repo\deepTimeSeries\data\datetime.npy'))
    date_list = np.array(
        [datetime.datetime.strftime(dt, '%Y-%m-%d')[5:] for dt in pd.date_range('2018-3-1', '2018-4-19').date])

    missing_rate = np.where(mask == 1, 0, 1).sum(axis=1) / mask.shape[1]
    # print((missing_rate[:, 0] != missing_rate[:, 1]).sum())
    plt.figure(figsize=(6, 10))

    for i, title in enumerate(['Missing rate of upstream', 'Missing rate of downstream']):
        plt.subplot(2, 1, i + 1)
        sns.heatmap(missing_rate[:, i].reshape((50, 24)), cmap='YlGnBu', linewidths=0.5)
        plt.xticks(ticks=range(0, 24, 2), labels=range(0, 24, 2), rotation=0)
        plt.yticks(ticks=range(0, 50, 5), labels=date_list[range(0, 50, 5)], rotation=0)
        plt.xlabel('Hours')
        # plt.ylabel('\n'.join(['D', 'a', 't', 'e', 's']), rotation='horizontal', verticalalignment='bottom',
        #            horizontalalignment='right')
        plt.ylabel('Dates', rotation=0, horizontalalignment='right')
        plt.title(title + ' traffic changes with time')
    plt.savefig(r'C:\Users\DengTao\Desktop\明理杯\missing_rate.jpg', dpi=150, bbox_inches='tight')


def plot_time_series(index, days):
    X = np.load(r'F:\git_repo\deepTimeSeries\data\X_filled_20.npy').swapaxes(1, 2)
    mask = np.load(r'F:\git_repo\deepTimeSeries\data\mask_20.npy')
    X = X * mask  # 3 8 16 18
    data = pd.DataFrame(data=X[:24 * days, index, :], columns=['Upstream traffic', 'Downstream traffic'])
    sns.lineplot(data=data, style_order=['Downstream traffic', 'Upstream traffic'])
    plt.xticks(ticks=range(0, 24 * days + 1, 6), labels=[0] + [6, 12, 18, 24] * days)
    plt.xlabel("Time (h)")
    plt.ylabel("Traffic volume (GB)")
    plt.legend()
    plt.savefig(r'C:\Users\DengTao\Desktop\明理杯\single_ts.jpg', dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)
    sns.set(font=myfont.get_name())
    mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # plot_missing_rate()
    # plot_miner_result()
    # plot_origin_result()
    plot_time_series(index=16, days=5)