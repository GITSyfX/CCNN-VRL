import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import numpy as np
from package import datap
from scipy.ndimage import gaussian_filter1d
import importlib
importlib.reload(stats)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'regular'

def make_wide(df, varname):
    sub = df[['subjnum', 'group', 'stage_label', varname]]
    wide = sub.pivot_table(index=['subjnum','group'],
                           columns='stage_label',
                           values=varname).reset_index()
    wide.columns.name = None
    return wide.rename(columns={'pre':f'{varname}_pre',
                                'post':f'{varname}_post',
                                'follow-up':f'{varname}_followup'})

def basic_format(ax, x_name, y_name, legend='off', linewidth=1.5,
                multiplots = None, ratio=None):
    
    if legend == 'on':
        ax.legend(loc='lower left', bbox_to_anchor=(0.7, 1), frameon=False, fontsize = 10)
    elif legend == 'off' and ax.get_legend() is not None:
        ax.get_legend().remove()

    if multiplots == None:
        ax.set_position([0.2, 0.2, 0.6, 0.6])
        
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.tick_params(axis='both', which='major', width=linewidth, length=7)
    ax.tick_params(axis='both', which='minor', width=linewidth, length=7)

    # 去除边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel(y_name, fontsize=14)
    ax.set_xlabel(x_name, fontsize=14)
    ax.set_box_aspect(ratio)        
    ax.tick_params(axis='both', labelsize=14)
    ax.set_axisbelow(False)

    

def violin(ax, data, x, y, order = None, palette = None, color = None, orient='v',
        hue=None, hue_order=None, clip_on = True,
        mean_marker_size=6, err_capsize=.11, scatter_size=5):

        if hue is not None and hue_order is None:
            hue_order = sorted(data[hue].dropna().unique())

        g_var = y if orient=='h' else x
        v_var = x if orient=='h' else y
        v=sns.violinplot(data=data, 
                            x=x, y=y, order=order, 
                            hue=hue, hue_order=hue_order, 
                            orient=orient, palette=palette, color=color, 
                            alpha=.1, inner=None, density_norm='width',
                            legend=False, clip_on=clip_on,
                            ax=ax)
        plt.setp(v.collections, alpha=.25, edgecolor='none')
        sns.stripplot(data=data, 
                            x=x, y=y, order=order, 
                            hue=g_var if hue is None else hue, 
                            hue_order=order if hue is None else hue_order, 
                            orient=orient, palette=palette, color=color, 
                            size=scatter_size,
                            edgecolor=None, jitter=True, alpha=.9,
                            dodge=False if hue is None else True,
                            zorder=2, clip_on=False,
                            ax=ax)
        sns.barplot(data=data, 
                            x=x, y=y, order=order, 
                            orient=orient, 
                            hue=hue, hue_order=hue_order,
                            errorbar='sd', linewidth=1, 
                            edgecolor=(0,0,0,0), facecolor=(0,0,0,0),
                            capsize=err_capsize, legend=False,
                            ax=ax)

        groupby = [g_var, hue] if hue is not None else [g_var]
        sns.stripplot(data=data.groupby(by=groupby)[v_var].mean().reset_index(), 
                        x=x, y=y, order=order, 
                        hue=hue, hue_order=hue_order, 
                        palette=[[.2]*3]*len(hue_order) if hue is not None else None,
                        dodge=False if hue is None else True, 
                        marker='o', size=mean_marker_size, color=[.2]*3, legend=False, ax=ax)

def viz_curve(ax, data, x, y , smooth = None, hue = None, hue_order=None, orient='x',
              palette = None):
    
    '''
    smooth:
        None: 不平滑
        True: 平滑
    '''
    data_plot = data.copy()
    y_col = y

    if hue is not None and hue_order is None:
        hue_order = sorted(data_plot[hue].dropna().unique())

    if smooth:
        # 创建一个新列存平滑后的值
        y_smooth = y + '_smooth'
        data_plot[y_smooth] = data_plot[y].values  # 先复制一份
        if hue is None:
            data_plot = data_plot.sort_values(by=x)
            data_plot[y_smooth] = gaussian_filter1d(data_plot[y_smooth].values, sigma=2)
        else:
            smoothed = []
            for g in hue_order:
                sub = data_plot[data_plot[hue] == g].copy()
                sub = sub.sort_values(by=x)
                sub[y_smooth] = gaussian_filter1d(sub[y].values, sigma=2)
                smoothed.append(sub)
            data_plot = pd.concat(smoothed, ignore_index=True)
        y_col = y_smooth

    sns.lineplot(
        data=data_plot, x=x, y=y_col,
        hue=hue, hue_order = hue_order, orient=orient,
        estimator="mean", errorbar=('ci', 95), palette=palette,
        linewidth=3,
        ax = ax)

def viz_corr(ax, data, x, y, color=None):
    sns.regplot(
    data=data, x=x, y=y,
    scatter_kws={"s": 40, "alpha": 0.7, "clip_on": False},
    line_kws={"linewidth": 3}, ci=95, color=color,
    ax=ax)

    df = data[[x, y]].dropna()
    r, p = stats.pearsonr(df[x], df[y])

    textstr = f"r = {r:.3f}\np = {p:.3f}"

    # 在右上角添加文字（坐标用轴的 fraction）
    ax.text(
        0, 1.05, textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=None
    )
    

if __name__ == '__main__':
    ## STEP 0: COLOR SETTING  
    group_color = [[44/255,135/255,132/254],
                [150/255,182/255,216/255],
                [151/255,200/255,175/255]]
    
    ## STEP 1: LOAD OR SAVE DATA    
    cfg = datap.load_config()
    dir = cfg["data_dir"]
    data_name = 'allfitdata_summary.csv'
    
    volatile_prob = [0.2, 0.8, 0.2, 0.8]
    last10_prob = 0.2

    # ---- 使用 numpy 拼接 ----
    ref = np.concatenate([
        np.full(90, 0.75),                      # 0–91 trials
        np.concatenate([np.full(20, v) for v in volatile_prob]),  # 4 × 20 block
        np.full(10, last10_prob)              # 最后10 trials
    ])


    summarydata = pd.read_csv(f'{dir}/{data_name}') 
    summarydata_pre = summarydata[summarydata['Stage label'] =='pre']
    summarydata_post = summarydata[summarydata['Stage label'] =='post']
    summarydata_followup = summarydata[summarydata['Stage label'] =='follow-up']
    
    summarydata_pre = summarydata_pre.reset_index(drop=True)
    summarydata_post = summarydata_post.reset_index(drop=True)
    summarydata_followup = summarydata_followup.reset_index(drop=True)

    summarydata_sham = summarydata[summarydata['Group label'] =='sham']
    summarydata_intervention = summarydata[summarydata['Group label'] =='intervention']


    ## STEP 2: PLOT DATA BY VIOLIN
    x = 'Stage label'
    y = 'Optimal choice rate'
    # fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    # violin(ax, summarydata, x=x, y=y, order=['pre','post','follow-up'],
    #     hue='Group label', palette=group_color)
    # ax.set_ylim([0, 5])
    # ax.set_yticks([0, 2.5, 5])
    # basic_format(ax,'',y, legend='on', ratio=0.8)
    # fig.savefig(f'Truecli_{x}_{y}.SVG', dpi=300)


    fig, axs = plt.subplots(1, 2, figsize=(7, 3), dpi=150)
    violin(axs[0], summarydata_intervention, x=x, y=y,
        order=['pre','post','follow-up'], palette=[group_color[1]]*3)
    # axs[0].set_ylim([0, 2])
    # axs[0].set_yticks([0, 1, 2])
    # axs[0].set_position([0.1, 0.2, 0.35, 0.6])
    basic_format(axs[0], '', y, legend='off', multiplots = True, ratio=0.8)

    violin(axs[1], summarydata_sham, x=x, y=y,
        order=['pre','post','follow-up'], palette=[group_color[2]]*3)
    # axs[1].set_ylim([0, 2])
    # axs[1].set_yticks([0, 1, 2])
    # axs[1].set_position([0.55, 0.2, 0.35, 0.6])
    basic_format(axs[1], '', y, legend='on', multiplots = True, ratio=0.8)
    fig.savefig(f'Truecli_{x}_{y}_bygroup.SVG', dpi=300)
    

    # STEP 3: PLOT DATA OF CORRELATION
    x = 'gamma'
    y = 'Visual crave'
    summarydata_diff =  summarydata_pre
    summarydata_diff[f'{x} diff'] = summarydata_post[x] - summarydata_pre[x]
    summarydata_diff[f'{y} diff'] = summarydata_post[y] - summarydata_pre[y]

    summarydata_diff = summarydata_diff[summarydata_diff['Group label'] == 'intervention']

    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    viz_corr(ax, summarydata_diff, x=f'{x} diff', y=f'{y} diff', color=group_color[1])
    basic_format(ax, f'{x} diff', f'{y} diff', legend='on', ratio=0.9)
    # ax.set_xlim([-1, 1])
    # ax.set_xticks([-1, 0, 1])
    # ax.set_ylim([-0.2, 0.2])
    # ax.set_yticks([-0.2, 0, 0.2])
    plt.grid(False)
    fig.savefig(f'Truecli_{x}diff_{y}diff_post-pre.SVG', dpi=300)
    plt.show()

