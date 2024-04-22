import numpy as np
import numpy.typing as npt
import joblib as jb
import logging as log
from itertools import permutations, combinations
from functools import partial

#p_l_t = jb.load('source/Gangelt_03_new_p_l_t.gz')


class P_ijT(object):
    def __init__(self, plt_filepath : str, ai_df_filepath : str, max_t = None,  agent_interactivity_dict = None) -> None:
        self.p_l_t = jb.load(plt_filepath)

        self.n_agents = self.p_l_t.shape[0]
        log.info(f'number of agents: {self.n_agents}')
        self.P_ijt_k = contact_lists_from_p_l_t(self.p_l_t, agent_interactivity_dict = agent_interactivity_dict,  max_t = max_t)   
        self.P_ijT = P_ijT_from_P_ijt_k(self.P_ijt_k, self.n_agents)
    
    def __call__(self) -> np.ndarray:
        return(self.P_ijT)

### Extracting the list of contacts l(list of tuples) for each timestep in one dictionary 
### missing - NPIs - how to excluded or replace locations 
def contact_lists_from_p_l_t(p_l_t: npt.NDArray, agent_interactivity_dict = None,  max_t = None, directed = True)->dict:
        """from the table (array) p_l_t  to the dictionary of lists of contacts 
        
        Args:
        P (np.arry): In the array  each entry represents a location ID and the first index agent IDs
        and the second index the time , 
        Kwargs:
        agent_interactivity_dict :dictionary for agent IDs and their respective interactivity
        usually  (default: None)
        max_t: How many timesteps are considered (default: None -> all in p_l_t)
        directed: Bool (default True) must be True for Matrix, False gives only unique pairs
        Returns:
        contacts :  dict of times with lists of contact tuples
        """
        
        """
        create a networkx Digraph (temporal ordered) from P_L_T Matrix
        if max_t is not specified it takes the max number of timesteps from the plt file
        """
        log.info('create contact list')
        contacts = {}
        n_agents = p_l_t.shape[0]

        if directed:
            iterfunc = partial(permutations, r=2)
        else:
            iterfunc = partial(combinations, r=2)

        if agent_interactivity_dict == None:
            agent_interactivity_dict = {a_ID: 1 for a_ID in range(n_agents)}
        
        if max_t== None:
            max_t=p_l_t.shape[1]
        
        for t in range(max_t):
            log.debug(f'{t}')
            # for all  locations occupied add directed edges for all inhabitants
            occupied_locs = np.unique(p_l_t[:, t])
            contacts[t] = []
            for loc in occupied_locs:
                agents_at_loc = np.where(p_l_t[:, t] == loc)
                n_agents_at_loc = len(agents_at_loc[0])
                if n_agents_at_loc > 1:  ## no additional self conections care for xx !!! todo
                    loc_weight = 1 / (
                            n_agents_at_loc - 1)  # np.log(1/(n_agents_at_loc-1))/np.log(log_base) # weights according to number at loc
                    contacts_list = [(x[0] , x[1] , loc_weight * agent_interactivity_dict[x[0]] * agent_interactivity_dict[
                            x[1]]) for x in iterfunc(agents_at_loc[0])] #(aID1,aID2,prob)
                else:
                    contacts_list =[]            
                contacts[t] += contacts_list
        log.info('contact list is done')               
        return contacts      

def get_lockdown_plt(old_plt : npt.NDArray, agents=None):
    """generates a lockdown plt from the input plt of the same size
        first column is used as lockdown location, usually the home location.
        
        If specified in the kwarg 'agents' the location are only changed
        for the agents with the IDs stated in the list.
        Otherwise the the lockdown is planned for all agents

    Args:
        old_plt (_type_): initial schedule as cols agent_IDs, rows: time (h), value: location ID
        agents (list, optional): list of agent IDs for lockdown

    Returns:
        np.array : array of shape of old_plt _description_
    """
    new_plt = old_plt.copy()
    timesteps = new_plt.shape[1] ## time 
    
    if agents:
        home_locs = old_plt[agents][:,0]
    else:
        home_locs = old_plt[:,0]
    
    new_plt[agents] = np.array([home_locs]*timesteps).T
    return new_plt

### go through dict and fill array or dict with += log(1-p_ij) 

def P_ijT_from_P_ijt_k(P_ijt_k: dict, n_agents: int):

    P_ijT = np.ones((n_agents-1,n_agents-1))
    
    for t, contacts in P_ijt_k.items():
        for a1,a2,p in contacts:
            P_ijT[a1-1,a2-1] *= (1-p)
    
    #log.info(f'Projection of P_ijt to PijT for {t} time steps is done')
          
    P_ijT = np.ones((n_agents-1,n_agents-1))-P_ijT

    return P_ijT 

def test_for_singularity_of_P(P: npt.NDArray)->tuple:
    """takes quadratic P matrix and returns singular P matrix and list of indeces with all zero entries in P

    Args:
        P (np.arry): 

    Returns:
        tuple : (P:np.array, zero_indeces: np.array)
    """
    indeces = np.where(P.any(axis=1))[0] ## indeces for cols and rows with at least one nonzero entry 

    if  np.linalg.det(P) == 0.0:
        log.info( 'P is a singular matrix, removing rows and columns with only zeros') 
        zero_contacts =  np.where(~P.any(axis=1))[0] ## indeces for the cols and rows with only zeros
        log.info(f'{len(zero_contacts)} agents have no certain contact in this time frame')
        P_n =P[np.ix_(indeces,indeces)] # np.ix_(indeces,indeces)==(indeces.reshape(len(indeces),1),indeces)
        return (P_n,zero_contacts, indeces)
    else:
        log.info('P is not singular')
        return(P, np.array([]), np.array([]))