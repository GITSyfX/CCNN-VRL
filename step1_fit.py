import os
import pickle
import numpy as np 
import multiprocessing as mp
from package import agent,datap,env,fit

eps_ = 1e-13


def get_pool(n_fit,n_cores):
    n = n_fit
    n_cores = n_cores if n_cores else int(mp.cpu_count()*.5) 
    print(f'Using {n_cores} parallel CPU cores\{n} ')
    return mp.Pool(n_cores)


if __name__ == '__main__':
    ## STEP 0: GET PARALLEL POOL
    n_fits = 80
    n_cores = 80
    mp.freeze_support()
    pool = get_pool(n_fits,n_cores)

    ## STEP 1: LOAD 

    cfg = datap.load_config()
    dir = cfg["data_dir"]
    agent_name = ['Model6','Model5','Model4','Model3','Model2','Model1','RA'] #

    stage_files = {
        "pre": f"{dir}/1vrl_pre_alldata.pkl",
        "post": f"{dir}/2vrl_post_alldata.pkl",
        "follow-up": f"{dir}/3vrl_follow-up_alldata.pkl"
    }

    ## STEP 2: SETTING 
    seed = 2025
    rng = np.random.RandomState(seed)

    ## STEP 3: FIT
    for stage, pkl_path in stage_files.items():
        agent_data = datap.load_pkl(os.path.dirname(pkl_path), os.path.splitext(os.path.basename(pkl_path))[0])
        for name in agent_name:
            task_agent = getattr(agent, name)
            all_results = fit.fl(pool,task_agent,agent_data,n_fits)


            output_path = f"{dir}/fitdata/fitresults_{name}_{stage}.pkl"
            with open(output_path, 'xb') as f:
                pickle.dump(all_results, f)

            
            print(f"已保存{stage}阶段, {name}模型结果 -> {output_path}")

    # summary the mean and std for parameters 
    pool.close()