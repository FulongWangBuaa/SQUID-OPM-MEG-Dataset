import numpy as np
import seaborn as sns
import pandas as pd

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')

def print_significance(p_value):
    if p_value < 0.0001:
        return "****"
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"
    
def half_violin(ax, data, pos, side='right', width=0.3, **kwargs):
    import scipy.stats
    # 计算核密度估计
    kde = scipy.stats.gaussian_kde(data)
    x = np.linspace(min(data), max(data), 100)
    y = kde(x)

    # 归一化y值
    y = y / y.max() * width

    if side == 'right':
        ax.fill_betweenx(x, pos, pos + y, **kwargs)
    else:
        ax.fill_betweenx(x, pos - y, pos, **kwargs)


def plot_cloudrain(data, ax, x_offset=0.25, colors=None, labels=None):
    '''
    绘制散点图和半小提琴图

    输入：
    data: 数据列表，每个元素是一个样本的数据
    ax: matplotlib.axes.Axes对象
    x_offset: x轴偏移量
    colors: 颜色列表，用于半小提琴图

    返回:
    ax: matplotlib.axes.Axes对象
    '''
    x_offset = 0.3
    if colors is None:
        colors = ['#007ACC','#E9657F','#2ECC40','#F1C40F','#E74C3C']

    for i,d in enumerate(data):
        half_violin(ax, d, i+x_offset, side='right', width=0.15,facecolor=colors[i], 
                    edgecolor=colors[i], alpha=0.8, linewidth=0.8)

    df_list = []
    for i, group in enumerate(data):
        for value in group:
            df_list.append({'Group': f'Group {i+1}', 'Value': value})
    df = pd.DataFrame(df_list)

    sns.stripplot(x='Group', y='Value', data=df, jitter=True, size=2.2, linewidth=0,palette=colors,alpha=0.8,ax=ax)
    ax.set_xlabel('')

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(0, len(medians))
    ax.scatter(inds+x_offset, medians, marker='o', color='white', s=5, zorder=3)
    ax.vlines(inds+x_offset, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds+x_offset, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    # ax.set_ylim(set_ylims[sub_idx])

    ax.set_xlim(0+x_offset/2-0.4, len(data)-1+x_offset/2+0.4)
    ax.set_xticks([i+x_offset/2 for i in range(len(data))])
    if labels is not None:
        ax.set_xticklabels(labels)

    return ax