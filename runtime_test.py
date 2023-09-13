import os
import sys
import yaml
import time
import copy


import GERDA_light as gl 
import matplotlib.pyplot as plt
import numpy as np
import logging as log
import matplotlib.pyplot as plt
import logging as log
import pandas as pd 
from functools import  partial
from multiprocessing.pool import Pool

logger = log.getLogger()
logger.setLevel(log.INFO)
logger.disabled = True

#plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-v0_8-muted')


def read_config_yml(config_file):
    
    '''
    load configuration file from 'analysis_config'.yaml**
    expected type: yaml
    '''
    assert (config_file in os.listdir()), f' could not find {config_file}' 
    log.info('use input_data/analysis/'+config_file+'.yaml as config file')

    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
    except:
        raise ImportError
    return config_dict
##without world generetion
def run_sim(k, opt, dT, timespan_day, world):
    #### generate simulation# ###
    t_ini0 = time.time()
    model = gl.SIS_model(world, sim_id=int(k))
    t_ini1 = time.time()
    ## initialize infection
    for i in np.arange(1,opt['n_ini_infected']+1):
        model.world.agents[i].state = 1 ## infect one agent
        model.world.agents[i].times['infection'] = 0    

    t_run0 = time.time()
    model.run(timespan=int(timespan_day*24/dT),**opt['run']) 
    t_run1 = time.time()
       
    ai_df =  model.world.ai_df
    ai_df['infection_day'] = ai_df[~ai_df['infection_time'].isna()]['infection_time'].map(lambda x: int(x*dT/24))
    inf_day_k =(k, list(ai_df['infection_day'].values))    
    times_k = (k,{'reset' : t_reset1 - t_reset0,  'run' : t_run1 - t_run0})
    return inf_day_k, times_k

### with world generation
def run_sim_2(k, opt, dT, timespan_day):
    t_w0= time.time()
    world = gl.World(dT=dT, **opt['world'] )
    t_w1= time.time()
        
    t_ini0 = time.time()
    model = gl.SIS_model(world, sim_id=int(k))
    t_ini1 = time.time()

    ## initialize infection
    for i in np.arange(1,opt['n_ini_infected']+1):
        model.world.agents[i].state = 1 ## infect one agent
        model.world.agents[i].times['infection'] = 0    

    t_run0 = time.time()
    model.run(timespan=int(timespan_day*24/dT),**opt['run']) 
    t_run1 = time.time()
       
    ai_df =  model.world.ai_df
    ai_df['infection_day'] = ai_df[~ai_df['infection_time'].isna()]['infection_time'].map(lambda x: int(x*dT/24))
    inf_day_k =(k, ai_df['infection_day'].values)    
    times_k = (k,{'world' : t_w1 - t_w0,'ini' : t_ini1 - t_ini0,  'run' : t_run1 - t_run0})
    del(model)
    del(world)
    return inf_day_k, times_k

if __name__ == '__main__':

    #### read config file ####
    try:
        config_file_name=sys.argv[1]
    except:
        config_file_name =  'runtime_test.yaml'
        log.info(f'no config filename was given. Using {config_file_name} instead')

    opt = read_config_yml(config_file_name)
    n = opt['n']
    dT = opt['dT']
    timespan_day = opt['timespan_day']
    name = opt['name']
    extended_name = name + f'_n_{n}_dT_{dT}_dur_{timespan_day}'
    inf_day_dict = {}
    times_dict = {}


    ##### load world ######
    #world = gl.World(dT=dT, **opt['world'] )
        
    #### run simulation #####
    #pre_sim = partial(run_sim, opt=opt, dT=dT, timespan_day=timespan_day, world=world)
    pre_sim2 = partial(run_sim_2, opt=opt, dT=dT, timespan_day=timespan_day)
    print('starting parallel simulations')
    #out_list = []
    with Pool(processes=n) as pool:
        out_list = pool.map(pre_sim2,range(n))
    #for k in range(2):
        #out_list.append(pre_sim(k))
    
    inf_day_dict = {x[0]: x[1] for x,y in out_list}
    times_dict = {y[0]: y[1] for x,y in out_list}
    
    #### create combined dataframes ### 
    out_df = pd.DataFrame(inf_day_dict)
    out_runtimes_df = pd.DataFrame(times_dict)
    print(f'average runtimes: {out_runtimes_df.mean(axis=1)}')

    #### plot ####
    fig, ax = plt.subplots(1,1, figsize=(6,6), sharey=True)
    for sim_id in out_df.columns:
        out_df.hist([sim_id],alpha=0.3, ax = ax, bins=np.arange(0,timespan_day,4))
    
    ax.set_title(f'{name}')
    ax.set_ylabel('infection events')
    ax.set_xlabel('time, days')
    #plt.show()

    ##### store output ### 
    fig.savefig('plots/runtime_test.png', bbox_inches='tight')
    out_df.to_csv(f'inf_times_day_{extended_name}.csv')
    out_runtimes_df.to_csv(f'runtimes_{extended_name}.csv')


