import numpy as np
import joblib as jb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from itertools import permutations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from G_utils.p_matrix import P_ijT, test_for_singularity_of_P
import logging as log 


log.debug('Debug Logging is acitive')

# log.debug('This is a debug message')
# log.info('This is an info message')
# log.warning('This is a warning message')
# log.error('This is an error message')
# log.critical('This is a critical message')


class spec_clustering(object):
    def __init__(self, plt_filename = 'source/Gangelt_03_new_p_l_t.gz',
                       ai_df_filename = 'source/Gangelt_03_new_ai_df.gz',
                       max_t = 168,
                       certainty = 0.999, ## lower probabilities interaction probabilities are set to zero
                       L_type = 'L', ## either L, L_sym or L_rw
                       #k = 250,
                       n_clusters: int = 250, 
                       inertia_plot=False,
                       plot_eigen_values=False,
                       use_suggested_k=False,
                       indvid_clust_for_exclud_agent = True,
                       ) -> None:
        log.info(f'max_t {max_t}')
        log.info(f'certainty {certainty}')
        self.certainty = certainty
        self.n_clusters = n_clusters
        self.k = n_clusters
        self.L_type = L_type 
        self.max_t = max_t

        ##  p_ijT is the likelyhood that agent i and agent j meet at least once during T
        self.P_c = P_ijT(plt_filepath=plt_filename,ai_df_filepath=ai_df_filename, max_t=self.max_t)
        self.n_agents = self.P_c.n_agents
        self.p_l_t = self.P_c.p_l_t

        ### load agent_information data 
        self.ai_df = jb.load(ai_df_filename)[['home','h_ID','type','age']]
        ## add household_size
        self.ai_df['household_size'] = self.ai_df['home'].map(self.ai_df.groupby('home').count()['h_ID'])
        ## agent specific interaction modifier : set to 1
        self.agent_interactivity_dict = {a_ID: 1 for a_ID in range(self.n_agents)}

        ### P_matrix
        self.reduce_P()
        
        ## Laplacian and eigenvalues _ vectors
        self.L = get_Laplacian(self.P_red, L_type = self.L_type)
        self.vals, self.vecs = get_sorted_eigenvalues_vectors(self.L)

        ## determine the optimal k value from eigenvalues, depends on threshold 1 
        s_k = self.suggested_k()
        if use_suggested_k:
            log.info(f'use suggested k:{s_k} instead of {n_clusters}')
            self.k = s_k
            self.n_clusters = s_k
        else:    
            log.info(f'suggested k is {s_k},Attentation! using  k= {n_clusters} instead')

        ## First K Eigenvectors 
        self.eigenvecs_df = get_first_k_eigenvecs(self.vals, self.vecs, k=self.k)
        log.info('eigenvalues done')
        self.cluster = run_k_means(self.eigenvecs_df, n_clusters=self.n_clusters)
        log.info('kmeans_done')

        ## cluster assignment
        self.clus_dict = self.get_clus_dict(indiviual_cluster=indvid_clust_for_exclud_agent)
        self.ai_df['cluster']=self.ai_df['h_ID'].map(self.clus_dict)
        
        ##  silhoutte_score - measure for the cluster assignment
        self.silhouette_score = silhouette_score(self.P_red, self.cluster)

        if plot_eigen_values:
            self.plot_eigenvalues()
        if inertia_plot:
            self.plot_kmeans_inertia()


    def reduce_P(self,binary=False):
        if binary:
            ##  probabilities lower then 'certainty' -> interaction probabilities are set to 0, Rest to 1
            self.P = np.where(self.P_c()>=self.certainty,1.,0.) 
        else:
            ##  probabilities lower then 'certainty' -> interaction probabilities are set to 0
            self.P = self.P_c() * (self.P_c()>self.certainty) ##     
        ## check for non interactors and remove them from P
        self.P_red, self.zero_contacts, self.non_zero_contacts = test_for_singularity_of_P(self.P) 

    def get_Laplacians(self):
        self.L = get_Laplacian(self.P_red, L_type = self.L_type)
        return self.L

    def suggested_k(self, diff=10, threshold=1):
        return np.argmax(np.diff(self.vals,diff)>threshold)

    def plot_eigenvalues(self):
        plt.plot(self.vals)
        plt.xlabel('index'), plt.ylabel('eigenvalue')
        plt.show()

    def plot_kmeans_inertia(self):
        inertias = []
        k_candidates = range(1, self.n_clusters)
        for k in k_candidates:
            k_means = KMeans(random_state=42, n_clusters=k)
            k_means.fit(self.eigenvecs_df)
            inertias.append(k_means.inertia_)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=k_candidates, y = inertias, s=80, ax=ax)
        sns.lineplot(x=k_candidates, y = inertias, alpha=0.5, ax=ax)
        ax.set(title='Inertia K-Means', ylabel='inertia', xlabel='k')
        plt.show()

    def get_clus_dict(self, indiviual_cluster=False, re_indexing=True):

        ## preset cluster for agents - to account for agents neglected in the clustering
        if indiviual_cluster:
            ##  assign all agents to an individual cluster  with cluster = -ID
            clus_dict = {x :-x for x in range(self.P.shape[0]+1)}
        else:    
            ##  assign all agents to one cluster -1    
            clus_dict = {x :-1 for x in range(self.P.shape[0]+1)} 
        ## assign cluster    
        clus_dict.update(dict(zip(self.non_zero_contacts+1, self.cluster)))  ## +1 important for translation between np.array and h_ID in GERDA
        
        if re_indexing: ## cluster with negative numbers get positive incremental values
            clusters =list(set(clus_dict.values()))
            re_index_dict = {x:i for i,x in enumerate(clusters)}
            clus_dict  = {i:re_index_dict[c] for i,c in clus_dict.items()}
        return clus_dict

    def get_cluster_sizes(self):
        clusters_arr = np.array(list(self.clus_dict.values()))
        _, c = np.unique(clusters_arr, return_counts=True)
        return c    

    #def cluster_to_ai_df(self):
    #    self.ai_df['cluster']=self.ai_df['h_ID'].map(self.clus_dict)
        #self.ai_df['household_size'] = self.ai_df['home'].map(self.ai_df.groupby('home').count()['h_ID'])
        #self.ai_df['cluster_size'] = self.ai_df['cluster'].map(self.ai_df.groupby('cluster').count()['h_ID'])    

      

def get_Laplacian(P: np.ndarray, L_type ='L'):
    """Laplacian"""
    assert L_type in ['L','L_sym','L_rw'], 'L_type muss be either  "L", "L_sym", or "L_rw" '
    ##degree matrix of P
    
    D = np.diag(P.sum(axis=0))
    L = D - P     ## unnormalized Laplacian
    
    if L_type == 'L': 
        return L  ## unnormalized Laplacian
        
    elif L_type == 'L_sym':     ## normalized symmetric laplacian
        D12 =  np.diag(np.sqrt(1 / np.diag(np.linalg.inv(D)))) ##  $\mathcal{L_{sym}} = D^{-1/2} \mathcal{L} D^{-1/2}$
        L_sym =  D12.dot(L.dot(D12))
        return L_sym
    elif L_type == 'L_rw':       ## normalized random walk laplacian     
        L_rw = np.identity(P.shape[0]) - np.linalg.inv(D).dot(P)    ##  $ \mathcal{L_{rw}} = D^{-1}L = I-D^{-1}P$ I-D^-1 P 
        return L_rw         

def get_sorted_eigenvalues_vectors(L: np.ndarray):
    """Eigenvalues & Vectors"""
    eigenvals, eigenvecs =  np.linalg.eig(L)
    vals, vecs = np.real(eigenvals), np.real(eigenvecs)
    ind = np.argsort(vals) ## sort by ascending eigenvalues
    return  vals[ind], vecs[:,ind] #np.real(eigenvals), np.real(eigenvecs) ## only real parts 

def get_first_k_eigenvecs(vals: np.ndarray ,vecs: np.ndarray, k:int=250)-> pd.DataFrame:
    """First Eigenvectors in Dataframe """
    df = pd.DataFrame(vecs[:,:k]) ## df from eigenvec corresponding to the lowest k eigenvalues
    df.columns = ['v_' + str(c) for c in df.columns]
    return df 

def run_k_means(df, n_clusters):
    """K-means clustering."""
    k_means = KMeans(n_clusters=n_clusters,n_init=10, init="k-means++")
    k_means.fit(df)
    cluster = k_means.fit_predict(df)
    return cluster




if __name__=="__main__":
    log.basicConfig(level=log.INFO)
    SC = spec_clustering()
    












  