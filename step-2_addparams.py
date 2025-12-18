import pandas as pd
import pickle
from package import agent,datap

cfg = datap.load_config()
dir = cfg["data_dir"]
agent_name = ['Model9'] 
params_name = ['alpha','beta','omiga','kappa_stim']
stages = ['pre','post','follow-up']

datafile = f'{dir}/alldata_summary.csv' 
savefile = f'{dir}/allfitdata_summary.csv' 

block = 'stable'

df = pd.read_csv(datafile)
print(f"读取Excel文件，共 {len(df)} 行数据")

# 读取pkl文件

for i,stage in enumerate(stages):
        for i,name in enumerate(agent_name):
                task_agent = getattr(agent,name)
                with open(f'{dir}/fitdata/fitresults_{name}_{stage}.pkl', 'rb') as f: 
                        All_fitdata = pickle.load(f)

                All_fitdata = datap.load_data(task_agent,All_fitdata)

                for idx, row in All_fitdata.iterrows():
                        subjnum = row['subj_id']
                        
                        # 在df中找到对应的行（匹配Subjnum和Stage）
                        mask = (df['Subjnum'] == int(subjnum)) & (df['Stage label'] == stage)
                        
                        if mask.sum() == 0:
                                print(f"  警告: 未找到 Subjnum={subjnum}, Stage label'={stage} 的数据")
                                continue
                        elif mask.sum() > 1:
                                print(f"  警告: 找到多行 Subjnum={subjnum}, Stage label'={stage} 的数据")
                        
                        # 更新参数值
                        for param_name in params_name:
                                df.loc[mask, param_name] = row[param_name]
                        
                        print(f"{stage} 阶段处理完成")

# ========== 提取参数并添加到DataFrame ==========

# ========== 保存结果 ==========
df.to_csv(savefile, index=False)
print(f"结果已保存到 {savefile}")