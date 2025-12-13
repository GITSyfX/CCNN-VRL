import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from package import datap
from scipy.ndimage import gaussian_filter1d

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

def basic_format(ax, x_name, y_name, legend='off', linewidth=1.5, ratio=None):
    if legend == 'off':
        ax.legend_.remove()
    else:
        ax.legend(loc='lower left', bbox_to_anchor=(0.7, 1), frameon=False, fontsize = 12)
        
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.tick_params(axis='both', which='major', width=linewidth, length=7)
    ax.tick_params(axis='both', which='minor', width=linewidth, length=7)

    # 去除边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel(y_name, fontsize=12)
    ax.set_xlabel(x_name, fontsize=12)
    ax.set_box_aspect(ratio)
    ax.set_position([0.2, 0.2, 0.6, 0.6])
    ax.tick_params(axis='both', labelsize=12)

    

def violin(ax, data, x, y, order = None, palette = None, orient='v',
        hue=None, hue_order=None, 
        mean_marker_size=6, err_capsize=.11, scatter_size=5):

        if hue is not None and hue_order is None:
            hue_order = sorted(data[hue].dropna().unique())

        g_var = y if orient=='h' else x
        v_var = x if orient=='h' else y
        v=sns.violinplot(data=data, 
                            x=x, y=y, order=order, 
                            hue=hue, hue_order=hue_order,
                            orient=orient, palette=palette, 
                            alpha=.1, inner=None, density_norm='width',
                            legend=False, clip_on=False,
                            ax=ax)
        plt.setp(v.collections, alpha=.35, edgecolor='none')
        sns.stripplot(data=data, 
                            x=x, y=y, order=order, 
                            hue=g_var if hue is None else hue, 
                            hue_order=order if hue is None else hue_order, 
                            orient=orient, palette=palette, 
                            size=scatter_size,
                            edgecolor=None, jitter=True, alpha=.7,
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

    # ax.set_xlabel('Trials')
    # ax.set_ylabel(indicname)
    # ax.set_ylim(y_lim[0],y_lim[1])
    # ax.set_yticks(y_ticks)
    # ax.set_xlim(0,150)
    # ax.set_xticks([0,37,75,112,150])
    # ax.text(0.105, 0.1, 'HS',transform=ax.transAxes,fontsize=12, alpha = 0.5)
    # ax.text(0.355, 0.1, 'HF',transform=ax.transAxes, fontsize=12, alpha = 0.5)
    # ax.text(0.605, 0.1, 'LS',transform=ax.transAxes, fontsize=12, alpha = 0.5)
    # ax.text(0.855, 0.1, 'LF',transform=ax.transAxes, fontsize=12, alpha = 0.5)
    # basic_format(ax,'Trials',indicname)
    # fig.legend(frameon=False)
    # plt.show()

if __name__ == '__main__':
    ## STEP 0: COLOR SETTING  
    group_color = [[150/255,182/255,216/255],
                   [151/255,200/255,175/255]]
    
    ## STEP 1: LOAD OR SAVE DATA    
    cfg = datap.load_config()
    dir = cfg["data_dir"]
    trldata_name = 'alldata_trlbytrl.csv'
    summarydata_name = 'alldata_summary.csv'
    stage_map = {1: 'pre', 2: 'post', 3: 'follow-up'}
    group_map = {0: 'sham', 1: 'intervention'}
    
    volatile_prob = [0.2, 0.8, 0.2, 0.8]
    last10_prob = 0.2

    # ---- 使用 numpy 拼接 ----
    ref = np.concatenate([
        np.full(90, 0.75),                      # 0–91 trials
        np.concatenate([np.full(20, v) for v in volatile_prob]),  # 4 × 20 block
        np.full(10, last10_prob)              # 最后10 trials
    ])


    trlbytrldata = pd.read_csv(f'{dir}/{trldata_name}') 
    summarydata = pd.read_csv(f'{dir}/{summarydata_name}') 

    # RT_wide = make_wide(summarydata, 'RT')
    # logRT_wide = make_wide(summarydata, 'log RT')
    # Earn_wide = make_wide(summarydata, 'Earnings')
    # Hit_wide = make_wide(summarydata, 'Hit rate')
    # widebehavdata = (RT_wide.merge(logRT_wide, on=['Subjnum','Group']) 
    #         .merge(Earn_wide,  on=['Subjnum','Group'])
    #         .merge(Hit_wide,   on=['Subjnum','Group']))
    # summarydata.to_csv('behavdata_summary.csv', index=False)


    ## STEP 2: PLOT DATA BY BAR
    x = 'Stage label'
    y = 'Zeta'
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    violin(ax, summarydata, x=x, y=y, order=['pre','post','follow-up'],
        hue='Group label', palette=group_color)
    ax.set_ylim([0, 20])
    ax.set_yticks([0, 10, 20])
    basic_format(ax,'',y, legend='on', ratio=1)
    fig.savefig(f'Truebehav_{x}_{y}.jpg', dpi=300)


    ## STEP 3: PLOT DATA TRIAL BY TRIAL
    x = 'Trials'
    y = 'Learning rate3'
    stage = 'pre'
    stagedata = trlbytrldata[trlbytrldata['Stage label'] == stage]
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    # ax.plot(ref, linestyle="--", linewidth=2, color = 'black', alpha = 0.5)
    viz_curve(ax, stagedata, x=x, y=y,
            hue='Group label', palette = group_color)
    ax.set_xlim([1,180])
    ax.set_xticks([1, 45, 90, 135, 180])
    basic_format(ax, x, y, legend='on')
    fig.savefig(f'Fitparams_{x}_{y}_{stage}.jpg', dpi=300)
    plt.show()

