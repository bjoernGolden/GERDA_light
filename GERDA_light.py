from G_utils.p_matrix import contact_lists_from_p_l_t
from G_utils.spectral_clustering import spec_clustering
from G_utils.temporal_condensation import generate_condensed_inf_p_dict, approximate_PI
from scipy.stats import lognorm
from numpy.random import default_rng
from functools import partial  


import matplotlib.pyplot as plt
import numpy as np
import joblib as jb
import logging as log
import copy 
#import utils
from multiprocessing import Pool, set_start_method
try:
    set_start_method('fork')
except:
     pass

logger = log.getLogger()
logger.setLevel(log.INFO)

plt.style.use('dark_background')

## - comment,  # - hide lines


class Agent(object):
    def __init__(self,ID, state, size):
        self.ID = ID
        self.state = state
        self.times = {'infection': None}
        self.size = size


class World(object):
    def __init__(self, p_l_t_filepath = 'src/Gangelt_03_new_p_l_t.gz',
                       ai_df_filename = 'src/Gangelt_03_new_ai_df.gz',
                       dT = 1, ### need to be an integer and a divisor of 168 (schedule time span)
                       only_P1 = False, ### only for testing 
                       clustering = True,
                       k_I: float = 0.2 ,
                       infection_times_cluster_list=[0,1,2,2,3,3,3,4,4,4,5,5,5,5,5,6,6,6,6,6,7,7,7,7,8], ## list[0] must be 0
                       **cluster_kwargs :dict):
        self.infection_times_cluster_list = infection_times_cluster_list
        self.infect_prob_dist = create_lognorm_probability_dist(s=1,a=4, days=30) ## in days
        self.infect_prob_dist_per_size = get_infection_prob_dist_dict(s=1,a=4,
                                                                      infection_times_cluster_list=infection_times_cluster_list)
        self.global_inf = k_I
        self.clustering = clustering
        self.dT = dT
        self.only_P1 = only_P1 ### only for testing 

        if clustering:
            
            log.debug('world with clusters of agents')
            self.SC = spec_clustering(**cluster_kwargs,
                                      plt_filename = p_l_t_filepath,
                                      ai_df_filename = ai_df_filename,
                                      )

            self.p_l_t = self.SC.p_l_t
            self.ai_df = self.SC.ai_df
            self.hID_cID_dict = get_hID_cID_dict(self.SC)
            self.agent_contacts = contact_lists_from_p_l_t(self.p_l_t, directed=False)
            self.contacts = agent_contacts_to_cluster_contacts(self.agent_contacts,self.hID_cID_dict)
            self.add_sizes_to_ai_df()
            self.generate_agents(column = 'cluster')

        
        else:
            self.p_l_t = jb.load(p_l_t_filepath)
            self.ai_df = jb.load(ai_df_filename)[['home','h_ID','type','age']]
            log.debug('world with agents without clustering')
            self.SC = None 
            self.hID_cID_dict = {x: x for x in self.ai_df['h_ID'].to_list()} # each is their own cluster 
            self.ai_df['cluster'] = self.ai_df['h_ID']
            self.contacts = contact_lists_from_p_l_t(self.p_l_t, directed=False)
            self.add_sizes_to_ai_df()
            self.generate_agents(column = 'h_ID')

       
        self.max_cluster_size = self.ai_df['cluster_size'].max()
        log.info(f'max cluster size: {self.max_cluster_size}')

        self.n_agents = self.p_l_t.shape[0]-1 ## 0 is not an agent
        self.schedule_time_span = self.p_l_t.shape[1]
        
        if (self.dT > 0)&(not self.only_P1):## del only if testing is ok 
           assert self.schedule_time_span%self.dT==0 , ('dT musst be a devisor of the schedule time span (usually 168h)')
           self.contacts = generate_condensed_inf_p_dict(self.contacts, self.schedule_time_span, self.dT)

         
    def generate_agents(self, column = 'h_ID')->dict:
        if column== 'cluster':
            self.agents = {ID: Agent(ID, 0, size = self.ai_df['cluster_size'][self.ai_df[column]==ID].values[0]) for ID in self.ai_df[column].unique()}
        else:
            self.agents = {ID: Agent(ID, 0, size = 1) for ID in self.ai_df[column].unique()}     
       
    
    def initialize_infections(self, agents_to_infect=[1]):
        for hID in agents_to_infect:
            self.agents[hID].state = 1 # infect one agent
            self.agents[hID].times['infection'] = 0 

    ### dataframe manipulations
    def add_sizes_to_ai_df(self):
        self.ai_df['household_size'] = self.ai_df['home'].map(self.ai_df.groupby('home',group_keys=False).count()['h_ID'])
        self.ai_df['cluster_size'] = self.ai_df['cluster'].map(self.ai_df.groupby('cluster',group_keys=False).count()['h_ID'])



def determine_contact_pairs(cluster_contacts,t=1,seed=None, weekly_contacts=True)->list:
    rng = default_rng(seed=seed)
    if weekly_contacts:
        t1 = t%168
    else:
        t1=t    
    contact_cluster_pairs =[(x[0],x[1]) for x in cluster_contacts[t1] if x[2]>rng.random(1)]#ps[i]]
    log.debug(f'{len(contact_cluster_pairs)} contact pairs out of {len(cluster_contacts[t1])} at t={t1}')
    return contact_cluster_pairs

def agent_contacts_to_cluster_contacts(agent_contacts,hID_cID_dict)->dict:
    cluster_contacts = {} ### dict t: [(c1,c2,p),...]
    for t, c in agent_contacts.items():
        cluster_contacts[t] = [(hID_cID_dict[x-1],hID_cID_dict[y-1],p) for x,y,p in c if hID_cID_dict[x-1] != hID_cID_dict[y-1]]  ## agent ID transfomation -1!!!!
        ## only for logging
        inner_cluster_interactions = len([1 for x,y,p in c if hID_cID_dict[x-1] == hID_cID_dict[y-1]])
        log.info(f'inner cluster interactions {inner_cluster_interactions}')
    return cluster_contacts   

def get_hID_cID_dict(SC)->dict:
    ##  human ID to cluster ID dictionary
    SC.ai_df.cluster.unique()
    SC.ai_df.set_index('h_ID')
    return SC.ai_df['cluster'].to_dict()

class SIS_model(object):
    def __init__(self,world,sim_id=None, t=1, determine_inf_times_for_cluster=False):
        self.rng = default_rng(seed=sim_id)
        self.ID = sim_id
        self.world = copy.deepcopy(world)
        self.w0 = world 
        self.real_contacts = {}
        self.schedule_time_span = 168
        self.t = t
        ## if true  a homongeous model with n_agents = max cluster size is repeatedly run
        ## the average infection times are then used
        if determine_inf_times_for_cluster: 
            
            self.mean_inf_time = get_average_infection_times_mp(
                n_agents=int(self.world.max_cluster_size), n_samples=15, 
                t=600, k_I=self.world.global_inf , n_cores=4) ## t has to be dependent on the n_agents
            
            self.world.infect_prob_dist_per_size = get_infection_prob_dist_dict(
                s=1,a=4,
                infection_times_cluster_list=self.mean_inf_time)
        else:
            self.mean_inf_time = [0]# self.world.infection_times_cluster_list     
    
    def reset(self):
        size_dict = self.world.infect_prob_dist_per_size  
        self.world = copy.deepcopy(self.w0)
        self.world.infect_prob_dist_per_size = size_dict 
        self.t = 1
        del size_dict
        # self.__init__(self.w0,t=1)
        
    def run(self,timespan=96, 
            only_inf_rel_contacts: bool = True,
            size_dependent_inf_prob: bool = True,
            only_infection: bool = False):
        
        if size_dependent_inf_prob:
            log.debug(f'using cluster size dependent  infection probability:')
        
        t_start = self.t
        
        for t in range(t_start+1,t_start + timespan+1):# +1
            self.t = t ##update 
            st = self.t_to_schedule_t(t, dT = self.world.dT) # st schedule time or time of the week
            
            ## filter for infection relevant contacts only
            if only_inf_rel_contacts:
                contacts_pairs_at_t = [x for x in self.world.contacts[st] if set((self.world.agents[x[0]].state,self.world.agents[x[1]].state))=={0,1}] ## falsch!!!
                log.debug(f'{len(contacts_pairs_at_t)} infection relevant interactions out of {len(self.world.contacts[st])} total interactions')
                
            else:
                contacts_pairs_at_t = self.world.contacts[st]

            if only_infection: ### combined probability for contact and infection
                for triple in contacts_pairs_at_t: # (i,j,p_c)- pc can be a tuple ### could be parallel
                    self.infection_attempt_c(triple,size_dependent_inf_prob=size_dependent_inf_prob)

            else: ### separated probability for contact and infection
                contact_pairs = self.determine_contact_pairs(contacts_pairs_at_t)
                self.real_contacts[t] = contact_pairs
                for pair in contact_pairs:
                    self.infection_attempt(pair,size_dependent_inf_prob=size_dependent_inf_prob)


        ## write transtion times per cluster to ai_df  when run is finished
        transition_times = create_transition_times_dict(self.world.agents)
        self.write_cluster_times_to_ai_df(transition_times)
        del transition_times
        self.write_infection_times_per_indiviual()        
        
    
    
    def infection_attempt(self, pair: tuple, size_dependent_inf_prob=True):
        a1, a2 = self.world.agents[pair[0]], self.world.agents[pair[1]] 
        states = (a1.state, a2.state)
        
        if set(states)== {0,1}:
            agents = [[a1,a2][x] for x in states] ## sorting that agent[0] has state 0 and vice versa 
            ## infection probability
            inf_duration_days = int((self.t-agents[1].times['infection'])/24)
            if inf_duration_days < 21: ## replace 21 with length of distribution
                
                if size_dependent_inf_prob:
                    p_I = self.world.global_inf * self.world.infect_prob_dist_per_size[agents[1].size][inf_duration_days]
                else:    
                    p_I = self.world.global_inf * self.world.infect_prob_dist[inf_duration_days]
                
                if p_I > self.rng.random(1)[0]: ### rework
                    log.debug(f'p_I: {p_I}')
                    agents[0].state = 1  ## infected without preliminary, however p_I is 0 anyways for 1-2 days
                    agents[0].times['infection'] = self.t
    
    def infection_attempt_c(self, triple: tuple, size_dependent_inf_prob=True): 
        
        a1, a2, p_c = self.world.agents[triple[0]], self.world.agents[triple[1]], triple[2] 

        states = (a1.state, a2.state)
        
        if set(states)== {0,1}:
            agents = [[a1,a2][x] for x in states] ## sorting that agent[0] has state 0 and vice versa 
            ## infection probability
            inf_duration_days = int((self.t-agents[1].times['infection'])*self.world.dT/24)
            if inf_duration_days < 21: ## replace 21 with length of distribution
                
                p_I = self.get_pI(inf_duration_days, agents[1].size)
                
                if type(p_c) == tuple: 
                    P_I = approximate_PI(p_I, p_c)
                else:
                    P_I = p_I * p_c
                rand = self.rng.random(1)[0]
                if P_I > rand:#np.random.random():
                    #log.debug(f'P_I: {P_I}')
                    agents[0].state = 1  ## infected without preliminary, however p_I is 0 anyways for 1-2 days
                    agents[0].times['infection'] = self.t
                    #if agents[0].ID in [10,20,30,40]: #delme
                    #    print(f'sim {self.ID} infection of {agents[0].ID} at {self.t} with  PI: {P_I} >{rand}') ## delme

    def get_pI(self, infector_inf_duration_days, infector_agent_size):
        p_I = (self.world.global_inf * 
               self.world.infect_prob_dist_per_size[infector_agent_size][infector_inf_duration_days])
        return p_I
    
    def determine_contact_pairs(self,contact_list: list)->list:
        rng = default_rng(seed=self.ID)
        contact_pairs =[(x[0],x[1]) for x in contact_list if x[2]>rng.random(1)]
        log.debug(f'{len(contact_pairs)} contact pairs out of {len(contact_list)}')
        return contact_pairs
    
    def t_to_schedule_t(self, t:int, dT:int = 1)->int:
        return t%(int(self.schedule_time_span/dT))
    
    def write_infection_times_per_indiviual(self):
        self.world.ai_df = self.world.ai_df.groupby('cluster',group_keys=False).apply(assign_inf_timing, inf_timings=self.mean_inf_time).reset_index(drop=True)
        self.world.ai_df['infection_time'] = self.world.ai_df['cluster_infection_time']+self.world.ai_df['Infection_timing_in_cluster'] 
        ## todo infection time is not written if none is infected - change?  

    def write_cluster_times_to_ai_df(self, times_dict):
        'add colums for transition times per cluster to ai_df'
        for transition in times_dict:
            self.world.ai_df['cluster_'+transition+'_time'] = self.world.ai_df['cluster'].map(times_dict[transition])    
    

### utility function
def create_lognorm_probability_dist(s=1, ## sigma of lognorm
                                    a=4, ## mean of lognorm
                                    days=30):
    x = np.linspace(0,days+1,days+1)
    dist=lognorm(s,loc=a)
    pdf = dist.pdf(x)
    p = pdf/pdf.sum()
    log.debug(f'max probability  at day  {np.argmax(p)}')
    return p

def get_infection_prob_dist_dict(infection_times_cluster_list: list=[0,2,2,2,2,3,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,7],
                                 s=1, ## sigma of lognorm
                                 a=4, ## mean of lognorm
                                 )-> dict:
    """creates average infection probability distribution for different cluster sizes based on given infection times.

    Args:
        infection_times_cluster_list (list, optional): ordered list of infection times inside the cluster
        Defaults to [0,2,2,2,2,3,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,7].

    Returns:
        dict: probability dist per cluster size 
    """
    max_inf_time = max(infection_times_cluster_list)
    
    if max_inf_time > 10:
        days = max_inf_time + 10
    else:
        days = 30    

    p = create_lognorm_probability_dist(s=s,a=a,days=days)
    prob_per_size_dict = {1: p}
    max_cluster_size = len(infection_times_cluster_list)
    
    for i in range(1,max_cluster_size + 1):
        ps = [np.array(int(infection_times_cluster_list[k])*[0.0]+list(p[:-(int(infection_times_cluster_list[k]))])) for k in range(1,i)]
        prob_per_size_dict[i] = sum([p] + ps)/i
    return prob_per_size_dict 

##### get the timing of mean infection times inside the clusters

### create plt and ai_df  for n_agents in one location to create infection times
def create_homogenous_world(n_agents=10, k_I=0.2):
    h_w  = World(clustering=False,k_I=k_I,only_P1=True)
    df = h_w.ai_df#.drop(h_w.ai_df[h_w.ai_df['h_ID']>10])
    h_w.ai_df = df[df['h_ID']<n_agents+1]
    h_w.p_l_t = np.ones(shape=(n_agents+1,168)).astype(int) ## first line is no agent ->0 
    h_w.p_l_t[0] = 0 * h_w.p_l_t[0] ## first line is no agent ->0 
    h_w.contacts = contact_lists_from_p_l_t(h_w.p_l_t, directed=False)
    h_w.generate_agents(column = 'h_ID')
    h_w.n_agents=h_w.p_l_t.shape[0]-1
    h_w.schedule_time_span = h_w.p_l_t.shape[1]
    h_w.initialize_infections([1])
    return h_w 


def average_lists(t_lists: list)->list: 
    ## required since the lists have not the same length
    max_len = max([len(lst) for lst in t_lists])
    t_array = np.full((len(t_lists), max_len), np.nan)
    for i, lst in enumerate(t_lists):
        t_array[i, :len(lst)] = lst
    mean_inf_times = np.nanmean(t_array, axis=0).astype(int)
    return mean_inf_times


def run_single_simulation_for_inf_times(w,sim_id, t=600):
        model_t = SIS_model(w, determine_inf_times_for_cluster=False, sim_id=sim_id)
        model_t.run(timespan=t,only_inf_rel_contacts=True, size_dependent_inf_prob=False)
        times = [int(a.times['infection']/24) if a.times['infection'] is not None else np.nan for a in model_t.world.agents.values()]
        times.sort()
        log.info('run test for mean infection times')
        del model_t
        return(times)

def get_average_infection_times_mp(n_agents=12, n_samples=12, t= 600, k_I=0.2, n_cores=4):
    ## multi processing # seems to not converge 
    t_lists = list(np.arange(max(n_samples,n_agents)))
    log.debug('iter: ',len(t_lists))
    h_w = create_homogenous_world(n_agents=n_agents, k_I=k_I)
    
    f = partial(run_single_simulation_for_inf_times,h_w,t=t)
    with Pool(n_cores) as p:
          l = p.map(f, t_lists)
    return  average_lists(l)

def create_transition_times_dict(agents: dict) -> dict:
    times_dict = {}
    for transition in next(iter(agents.values())).times.keys():
        times_dict[transition] = {agent_id : agents[agent_id].times[transition] for agent_id in agents}
    return times_dict  

def write_cluster_times_to_ai_df(ai_df, times_dict):
    'add colums for transition times per cluster to ai_df'
    for transition in times_dict:
       ai_df['cluster_'+transition+'_time'] = ai_df['cluster'].map(times_dict[transition])

def assign_inf_timing(cluster, inf_timings, time_step_size=24):
    cluster['Infection_timing_in_cluster'] = np.array(inf_timings[:len(cluster)])*time_step_size 
    return cluster  

if __name__=="__main__":

    ## model generation and simulation
    w1 = World(clustering=False)
    model = SIS_model(w1)
    model.world.agents[1].state = 1 # infect one agent
    model.world.agents[1].times['infection'] = 0 
    model.run(timespan=1500, only_inf_rel_contacts=True)

    ## Data collection
    l1=[]
    l2 = []
    for id, a in model.world.agents.items():
        l1.append(a.times['infection'])
        l2.append(a.state)

    ### plots
    np.histogram(np.array([x for x in l1 if x is not None]))
    plt.hist(np.array([x for x in l1 if x is not None]))
    plt.xlabel('infection time')
    plt.ylabel('count')
    plt.show()