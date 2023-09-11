import logging as log 

def get_P1(p:list)-> float:
    return sum(p)

def get_P2(p:list)-> float: 
    ## assumed that p is arranged from early to later time points
    P2 = 0.0
    for l,pt in enumerate(p):
        P2 += pt * sum(p[:l]) ## numpy?
        #log.debug((P2,pt,l,p[:l]))
    return P2


def get_P3(p) -> float:
    P3 = 0.0
    for t, pt in enumerate(p):
        for l, pl in enumerate(p[:t]):
                P3 += pt * pl * sum(p[:l])  
    return P3        

def get_P4(p) -> float:
    P4 = 0.0
    for t, pt in enumerate(p):
        for l, pl in enumerate(p[:t]):
            for k, pk in enumerate(p[:l]):
                P4 += pt * pl * pk * sum(p[:k])  
    return P4  

def approx_4_PI(p,k_I):
    P1 = get_P1(p)
    P2 = get_P2(p)
    P3 = get_P3(p)
    P4 = get_P4(p)
    return k_I * P1 - (k_I ** 2) * P2 + (k_I ** 3) * P3  - (k_I ** 4) * P4 

def resort_contacts_for_agent_pairs(contacts)->dict:
    '''
    {t:[(i,j,p_i_j_t),]} -> {(i,j) : [(t1,p1),(t2,p2)]}
     resorting {t:[(i,j,p_i_j_t),]} -> {(i,j) : [(t1,p1),(t2,p2)]}
     since its sparse use tuples instead of list [p1,p2,... ]
     p1 contact 
    '''
    pair_dict = {} 
    for t, contact_probs_at_t in contacts.items():    
        for x in contact_probs_at_t:
            if (x[0],x[1]) in pair_dict:
                pair_dict[(x[0],x[1])].append((t,x[2]))
            else:
                pair_dict[(x[0],x[1])] = [(t,x[2])]
    return pair_dict


def generate_condensed_inf_p_dict(contacts, schedule_time_span,dT):
    '''
    dict {t:[(i,j,p_i_j_t),]} -> {T:[(i,j,(P1,P2,P3)]}
    '''
    
    T_max = int(schedule_time_span/dT)

    log.debug(f'generating the dict of the condensed time steps: T_max: {T_max}')

    pair_dict = resort_contacts_for_agent_pairs(contacts)

    new_time_dict = {}
    for  (i,j), t_p_list in pair_dict.items():

        t_p_T_list = [x+(int(x[0]/dT),) for x in  t_p_list]
    
        if  not t_p_T_list:
            continue ### if list is empty dont start the calculation
    
        for T in range(0,T_max+1): ## 0? or 1?
            p_list = [x[1] for x in t_p_T_list if x[2] == T]
        # T+1 ? 
            if p_list:
                P1 = get_P1(p_list)
                P2 = get_P2(p_list)
                P3 = get_P3(p_list)
                if P1 > 0:
                    if T in new_time_dict:
                        new_time_dict[T].append((i,j,(P1,P2,P3)))
                    else:
                        new_time_dict[T] =[(i,j,(P1,P2,P3))]
    return new_time_dict  

def approximate_PI(p_I:float, p_c: tuple):
    if len(p_c)==3:
        P1 = p_c[0]
        P2 = p_c[1]
        P3 = p_c[2]
        return p_I * P1 - (p_I ** 2) * P2 + (p_I ** 3) * P3
    else:
        raise TypeError('need 3 probabilites')



if __name__=='__main__':
    p_list = [0.3,0.4,0.5,0.6]
    log.debug(f'P1 : {get_P1(p_list)}')
    log.debug(f'P2 : {get_P2(p_list)}')
    log.debug(f'P3 : {get_P3(p_list)}')
    log.debug(f'P4 : {get_P4(p_list)}')