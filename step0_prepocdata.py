import os
import pandas as pd
from package import datap



if __name__ == "__main__":

    cfg = datap.load_config()
    dir = cfg["data_dir"]
    stages = ['pre','post','follow-up']
    for i,stage in enumerate(stages):
        

        behav_path = os.path.join(dir, "preprocdata", f"{i+1}vrl_{stage}")
        behav_pkl = os.path.join(dir, f"{i+1}vrl_{stage}_alldata.pkl")


        # 用于存储所有被试数据
        behav_alldata = {}

        # 遍历文件夹下的所有xlsx文件
        for file in os.listdir(behav_path):
            if file.endswith(".csv"):
                file_path = os.path.join(behav_path,file)
                df = pd.read_csv(file_path)
                # 去掉扩展名
                filename_no_ext = os.path.splitext(file)[0]
                
                subject_id = filename_no_ext.split("_")[0]  # 提取 ID 部分
                behav_alldata[subject_id] = df
        # 保存为pkl文件
        pd.to_pickle(behav_alldata, behav_pkl)
        print(f"所有被试: {len(behav_alldata)} 个被试，已保存到 {behav_pkl}")