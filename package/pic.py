import numpy as np
import matplotlib.pyplot as plt

def set_format(ax, x_name, y_name):
    font = {'family': 'Arial', 'weight': 'regular'}
    plt.rc('font', **font)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.tick_params(axis='both', which='major', width=1)
    ax.tick_params(axis='both', which='minor', width=1)

    # 去除边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel(y_name,fontsize=10, fontdict=font)
    plt.yticks(fontsize=10)
    ax.set_xlabel(x_name, fontsize=10, fontdict=font)
    plt.xticks(fontsize=10) 

def set_ticks(value, ax=None, lb=None, ub=None, whichaxis='y', 
               ticksnum=5, precision=None, delticks=None, pad_ratio=0.0):
    if ax is None:
        ax = plt.gca()

    value = np.array(value)
    d = np.abs(np.max(value) - np.min(value))

    vmax = np.max(value) + 0.1 * d
    vmin = np.min(value) - 0.1 * d

    # 自动推断精度
    if precision is None:
        digit_count = int(np.floor(np.log10(np.abs(vmax - vmin)))) + 1
        precision = 10 ** (digit_count - 1)

    # 计算上下界
    v_upper = np.ceil(vmax / precision) * precision if ub is None else ub
    v_bott  = np.floor(vmin / precision) * precision if lb is None else lb

    # 计算 ticks
    tick_interval = (v_upper - v_bott) / (ticksnum - 1)
    ticks = [v_bott + i * tick_interval for i in range(ticksnum)]

    # 删除指定位置的 ticks
    if delticks is not None:
        ticks = [t for i, t in enumerate(ticks) if i not in delticks]

    # padding 处理：只改坐标范围，不改 ticks
    pad_x = (ticks[-1] - ticks[0]) * pad_ratio
    pad_y = (ticks[-1] - ticks[0]) * pad_ratio

    if whichaxis == 'y':
        ax.set_ylim(ticks[0] - pad_y, ticks[-1] + pad_y)
        ax.set_yticks(ticks)
    elif whichaxis == 'x':
        ax.set_xlim(ticks[0] - pad_x, ticks[-1] + pad_x)
        ax.set_xticks(ticks)

    return ticks