import os
import sys
import yaml
import time
import copy


import GERDA_light as gl 
import matplotlib.pyplot as plt
import numpy as np
import logging as log
import seaborn as sns
import pandas as pd 
from functools import  partial
from multiprocessing.pool import Pool
from G_utils.p_matrix import get_lockdown_plt, contact_lists_from_p_l_t

logger = log.getLogger()
logger.setLevel(log.INFO)
logger.disabled = True

plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-v0_8-paper')


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

### not used here
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

def run_sim_lockdown(k:int, opt: dict):
    """function to simulate a lockdown to use in multiprocessing

    Args:
        k (int): _description_
        opt (dict): all world parameter are set in lockdown_test.yaml
        dT (int):temporal condensation range between 1 and 48 
        timespan_day (int): simulation time in days
        world (object): GERDA light world object

    Returns:
        tuple of tuples: ((simulation ID:int, infection times (days):list) , (simulation ID, runtimes: dict))
    """
    dT = opt['dT'] # temporal condensation range in h
    
    ### simulation times
    T1, T2 = int(opt['pre_lockdown_time']/dT), int(opt['lockdown_time']/dT) 
    T3 = int((opt['simulation_time']- T1 -T2)/dT) 
    
    
    #### generate simulation# ###
    t_ini0 = time.time()
    model = gl.SIS_model(world, sim_id=int(k))
    t_ini1 = time.time()
    
    ## initialize infection
    for i in opt['ini_infected']:
        model.world.agents[i].state = 1 ## infect one agent
        model.world.agents[i].times['infection'] = 0    

    ## pre lockdown 
    model.run(timespan=T1, only_inf_rel_contacts=True, only_infection=True)

    ## lockdown
    model.world.contacts = lockdown_contacts
    model.run(timespan=T2, only_inf_rel_contacts=True, only_infection=True)

    ## post lockdown
    model.world.contacts = contacts
    model.run(timespan=T3, only_inf_rel_contacts=True, only_infection=True)
    ai_df = model.world.ai_df
    ai_df['infection_day'] = ai_df[~ai_df['infection_time'].isna()]['infection_time'].map(lambda x: int(x*dT/24))
    inf_day_k =(k, list(ai_df['infection_day'].values))   
    AR_k = (k, np.sum([~pd.isna(ai_df.infection_day)])/world.n_agents)
    return inf_day_k, AR_k

### with world generation### not used here
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
        config_file_name =  'lockdown_test.yaml'
        log.info(f'no config filename was given. Using {config_file_name} instead')

    opt = read_config_yml(config_file_name)
    n = opt['n']
    dT = opt['dT']
    sim_time = opt['simulation_time']# timespan_day = opt['timespan_day']
    T1 = opt['pre_lockdown_time']
    T2 = opt['lockdown_time']
    name = 'lockdown_'+opt['name']
    extended_name = name + f'_n_{n}_dT_{dT}_dur_{sim_time}_T1_{T1}_T2_{T2}'
    inf_day_dict = {}
    times_dict = {}


    ##### world ######
    t_w0= time.time()
    world = gl.World(dT=dT, **opt['world'] )
    t_w1= time.time()
    contacts = world.contacts.copy()

    ### generate lockdown plt
    ## create new contact dict for new schedule (lockdown)
    new_plt = get_lockdown_plt(world.p_l_t)
    lockdown_contacts =  contact_lists_from_p_l_t(new_plt, directed=False) 
    lockdown_contacts = gl.generate_condensed_inf_p_dict(lockdown_contacts, 168, dT)
        
    #### run simulation #####
    #pre_sim = partial(run_sim, opt=opt, dT=dT, timespan_day=timespan_day, world=world)
    sim_partial = partial(run_sim_lockdown, opt=opt)
    #pre_sim2 = partial(run_sim_2, opt=opt, dT=dT, timespan_day=timespan_day)
    print('starting parallel simulations')
    #out_list = []
    with Pool(processes=n) as pool:
        out_list = pool.map(sim_partial,range(n))
    #for k in range(2):
        #out_list.append(pre_sim(k))
    
    inf_day_dict = {x[0]: x[1] for x,y in out_list}
    AR_dict = {y[0]: y[1] for x,y in out_list}
    
    #### create combined dataframes ### 
    out_df = pd.DataFrame(inf_day_dict)
    out_AR_df = pd.DataFrame(AR_dict, index=[1])
    #print(f'average runtimes: {out_runtimes_df.mean(axis=1)}')

    #### plot ####
    fig, axes = plt.subplots(1,2, figsize=(6,4))
    bins = np.arange(0,int(opt['simulation_time']/24),4)
    for sim_id in out_df.columns:
        #out_df.hist([sim_id],alpha=0.3, ax = ax, bins=np.arange(0,int(opt['simulation_time']/24),4))
        y,x = np.histogram(list(out_df[sim_id][~pd.isna(out_df[sim_id])].values), bins=bins)
        axes[0].plot(x[:-1],y, alpha=0.1, color='red')
    
    axes[0].set_ylabel('infection events')
    axes[0].set_xlabel('time, days')
    axes[0].set_xlim(0,100)
    axes[0].set_ylim(-5,180)
    axes[0].set_title(f'{name}')


    df1 = out_AR_df.T.reset_index(drop=True)
    sns.swarmplot(data=df1, ax=axes[1])
    axes[1].set_ylabel('attack rate')

    plt.tight_layout()
    plt.show()

    ##### store output ### 
    fig.savefig('plots/lockdown_test.png', bbox_inches='tight')
    out_df.to_csv(f'output_server/hd/inf_times_day_{extended_name}.csv')
    out_AR_df.to_csv(f'output_server/hd/AR_{extended_name}.csv')


