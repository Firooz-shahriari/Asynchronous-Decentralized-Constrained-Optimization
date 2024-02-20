import numpy as np
import matplotlib.pyplot as plt








def base_function(m, k, z_p):
    exponent = np.abs(k - m)
    sign = 1 if (k - m) >= 0 else -1
    return (z_p ** exponent) * sign

def term_1(m, K, z_p):
    return 2*sum([abs(base_function(m, k, z_p)) for k in K])

def term_2_3_inner_function(m, k1, z_p, other_ks, c):
    ## k1 (float): The k value for the first base function.
    ## z_p (float): A common z_p value.
    ## other_ks (list): A list of k values for the other base functions (which we are subtracting them from the first base function).
    ## cs (list): A list of scaling constants for the other base functions.
    first_term = base_function(m, k1, z_p)
    sum_of_terms = np.float64(c) * sum([base_function(m, k, z_p) for k in other_ks])
    return abs(first_term - sum_of_terms)
 
def term_2_3(m, k1_range, z_p, other_ks_list, c_list):    
    ## k1_range (list): A range of k1 values.
    return sum([term_2_3_inner_function(m, k1, z_p, other_ks, c) for k1, other_ks, c in zip(k1_range,other_ks_list,c_list)])

def base_function2(m, k, z_p):
    exponent = np.abs(k-1 - m)
    sign = 1 if (k-1 - m) >= 0 else -1
    return (z_p ** exponent) * sign

def term_2_inner_function(m, k1, z_p, other_ks, c):
    ## k1 (float): The k value for the first base function.
    ## z_p (float): A common z_p value.
    ## other_ks (list): A list of k values for the other base functions (which we are subtracting them from the first base function).
    ## cs (list): A list of scaling constants for the other base functions.
    first_term = base_function2(m, k1, z_p)
    sum_of_terms = np.float64(c) * sum([base_function2(m, k, z_p) for k in other_ks])
    return abs(first_term - sum_of_terms)
 
def term_2(m, k1_range, z_p, other_ks_list, c_list):    
    ## k1_range (list): A range of k1 values.
    return sum([term_2_inner_function(m, k1, z_p, other_ks, c) for k1, other_ks, c in zip(k1_range,other_ks_list,c_list)])


def plot_and_max(m_range, function):
    # Finding the maximum in the discrete range
    max_index = np.argmax(function)
    max_m_discrete = m_range[max_index]
    max_value_discrete = function[max_index]

    # Plotting discrete points
    %matplotlib inline
    plt.figure(figsize=(10, 6))
    plt.scatter(m_range, function, color='black', s=15)
    plt.scatter(max_m_discrete, max_value_discrete, color='red', s=20) # , label=f'Maximum at m={max_m_discrete}')
    # plt.legend()
    plt.grid(True)
    plt.show()

    print(max_m_discrete, max_value_discrete)



def compute_delay_matrix(seed, num_nodes, max_iter, delay_type):
    if delay_type == 'uniform:1-100':
        Delay_mat = np.random.uniform(1, 100, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-50':
        Delay_mat = np.random.uniform(1, 50, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-80':
        Delay_mat = np.random.uniform(1, 80, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-1':
        Delay_mat = np.random.uniform(1, 1, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-2':
        Delay_mat = np.random.uniform(1, 2, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-200':
        Delay_mat = np.random.uniform(1, 200, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-3':
        Delay_mat = np.random.uniform(1, 3, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-4':
        Delay_mat = np.random.uniform(1, 4, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-20':
        Delay_mat = np.random.uniform(1, 20, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-30':
        Delay_mat = np.random.uniform(1, 30, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-40':
        Delay_mat = np.random.uniform(1, 40, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-60':
        Delay_mat = np.random.uniform(1, 60, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-5':
        Delay_mat = np.random.uniform(1, 5, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-10':
        Delay_mat = np.random.uniform(1, 10, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-500':
        Delay_mat = np.random.uniform(1, 500, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-1000':
        Delay_mat = np.random.uniform(1, 1000, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'uniform:1-50':
        Delay_mat = np.random.uniform(1, 50, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'exp:scale:10':
        Delay_mat = np.random.exponential(10,   size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'no_delay':
        Delay_mat = np.zeros((num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'one_node_constant_delay_500':
        Delay_mat = np.zeros((num_nodes,num_nodes,max_iter+2))
        Delay_mat[:,0,:] = np.random.uniform(500,500,(num_nodes, max_iter+2))
    elif delay_type == 'all_nodes_constant_delay_100':
        Delay_mat = np.random.uniform(100, 100, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'all_nodes_constant_delay_500':
        Delay_mat = np.random.uniform(500, 500, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'all_nodes_constant_delay_1000':
        Delay_mat = np.random.uniform(1000, 1000, size=(num_nodes,num_nodes,max_iter+2))
    elif delay_type == 'counter_example_one_node_fails': ## all messages are lost
        Delay_mat = np.random.uniform(min_delay, max_delay, size=(num_nodes,num_nodes,max_iter+2))
        Delay_mat[:,0,:] = np.random.normal(1e10, 0, (num_nodes, max_iter+2))
    elif delay_type == '20percent_1_link_delay_10_others_ideal':
        Delay_mat = np.zeros((num_nodes,num_nodes,max_iter+2)) 
        np.random.seed(seed)
        rand_iters = np.random.randint(1, max_iter, size=int(max_iter/5))
        Delay_mat[:,0, rand_iters] = np.random.normal(10,0,(num_nodes,len(rand_iters)))
    elif delay_type == '20percent_1_link_delay_20_others_ideal':
        Delay_mat = np.zeros((num_nodes,num_nodes,max_iter+2)) 
        np.random.seed(seed)
        rand_iters = np.random.randint(1, max_iter, size=int(max_iter/5))
        Delay_mat[:,0, rand_iters] = np.random.normal(20,0,(num_nodes,len(rand_iters)))
    elif delay_type == '20percent_1_link_delay_40_others_ideal':
        Delay_mat = np.zeros((num_nodes,num_nodes,max_iter+2)) 
        np.random.seed(seed)
        rand_iters = np.random.randint(1, max_iter, size=int(max_iter/5))
        Delay_mat[:,0, rand_iters] = np.random.normal(40,0,(num_nodes,len(rand_iters)))
    elif delay_type == '20percent_1_link_delay_80_others_ideal':
        Delay_mat = np.zeros((num_nodes,num_nodes,max_iter+2))
        np.random.seed(seed)
        rand_iters = np.random.randint(1, max_iter, size=int(max_iter/5))
        Delay_mat[:,0, rand_iters] = np.random.normal(80,0,(num_nodes,len(rand_iters)))
    elif delay_type == '20percent_1_link_delay_160_others_ideal':
        Delay_mat = np.zeros((num_nodes,num_nodes,max_iter+2))
        np.random.seed(seed)
        rand_iters = np.random.randint(1, max_iter, size=int(max_iter/5))
        Delay_mat[:,0, rand_iters] = np.random.normal(160,0,(num_nodes,len(rand_iters)))
    elif delay_type == '20percent_1_link_delay_400_others_ideal':
        Delay_mat = np.zeros((num_nodes,num_nodes,max_iter+2))
        np.random.seed(seed)
        rand_iters = np.random.randint(1, max_iter, size=int(max_iter/5))
        Delay_mat[:,0, rand_iters] = np.random.normal(400,0,(num_nodes,len(rand_iters)))
    elif delay_type == 'normal_150_10':
        Delay_mat = np.random.normal(150,10, size=(num_nodes,num_nodes,max_iter+2))
        Delay_mat[Delay_mat < 0] = 0
    elif delay_type == 'normal_150_20':
        Delay_mat = np.random.normal(150,20, size=(num_nodes,num_nodes,max_iter+2))
        Delay_mat[Delay_mat < 0] = 0
    elif delay_type == 'normal_150_40':
        Delay_mat = np.random.normal(150,40, size=(num_nodes,num_nodes,max_iter+2))
        Delay_mat[Delay_mat < 0] = 0
    elif delay_type == 'normal_150_80':
        Delay_mat = np.random.normal(150,80, size=(num_nodes,num_nodes,max_iter+2))
        Delay_mat[Delay_mat < 0] = 0
    elif delay_type == 'normal_600_80':
        Delay_mat = np.random.normal(600,80, size=(num_nodes,num_nodes,max_iter+2))
        Delay_mat[Delay_mat < 0] = 0
    elif delay_type == 'normal_150_0':
        Delay_mat = np.random.normal(150,0,  size=(num_nodes,num_nodes,max_iter+2))
        Delay_mat[Delay_mat < 0] = 0
    range_iters = np.arange(max_iter)
    for i in range(max_iter):
        np.fill_diagonal(Delay_mat[:,:,i], 0)
    return Delay_mat[:,:,range_iters]



import matplotlib.pyplot as plt
import numpy as np
import inspect 
import os
import matplotlib
from matplotlib.font_manager import FontProperties



def plot_sorted(in_list, show=True):
    kappa_sorted = sorted(in_list)
    indexed_list = list(enumerate(in_list))
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1])
    sorted_indices = [index for index, value in sorted_indexed_list]

    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    var_names = [var_name for var_name, var_val in callers_local_vars if var_val is in_list]

    if show:
        plt.plot(np.arange(len(kappa_sorted)), kappa_sorted, marker='o')  # 'o' for circular markers
        ax = plt.gca()
        new_tick_positions = range(len(kappa_sorted))
        ax.set_xticks(new_tick_positions)
        ax.set_xticklabels(sorted_indices)  # Example labels

        plt.title(f'{var_names}')
        # plt.show()
    return sorted_indices

def compute_kappa_and_terms_max(delay_profiles, min_K, z_p, m_min, m_max, show=True ):
    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams['pdf.fonttype'] = 42
    font = FontProperties()
    font.set_size(17)
    font2 = FontProperties()
    font2.set_size(17)
    mark_every = 50000
    linewidth = 2

    kappa_list = []
    term1_list = []
    term2_list = []
    term3_list = []
    for my_delay in delay_profiles:
        max_K = len(my_delay)
        Delay_mat = np.zeros((4,4,len(my_delay)))
        Delay_mat[1,0,np.arange(len(my_delay))] = my_delay
        kappa, m_range, t1,t2,t3 = tau_vu_p_m(Delay_mat,1,0,min_K,max_K, z_p, m_min, m_max)
        # print(kappa)
        max_index_kappa = np.argmax(kappa)
        max_value__kappa = kappa[max_index_kappa]
        max_index_term1 = np.argmax(t1)
        max_value__term1 = t1[max_index_term1]
        max_index_term2 = np.argmax(t2)
        max_value__term2 = t2[max_index_term2]
        max_index_term3 = np.argmax(t3)
        max_value__term3 = t3[max_index_term3]

        kappa_list.append(max_value__kappa)
        term1_list.append(max_value__term1)    
        term2_list.append(max_value__term2)    
        term3_list.append(max_value__term3)

        # print(max_index_kappa)
        # print(max_value__kappa)
    
    # folder   = 'check_theorem_figure' 
    folder   = 'new_check_theorem_figure' 

    if not os.path.exists(folder):
        os.makedirs(folder)

    %matplotlib inline
    x_axis = np.arange(len(delay_profiles))+1
    plt.figure(figsize=(10, 6))
    plt.tick_params(labelsize=19, width=3)
    plt.scatter(x_axis, kappa_list, color='c', s=120, linewidth = linewidth)
    # plt.scatter(x_axis, term1_list, color='blue', s=15, label='max_term 1')
    # plt.scatter(x_axis, term2_list, color='green', s=15, label='max_term 2')
    # plt.scatter(x_axis, term3_list, color='pink', s=15, label='max_term 3')
    # # plt.scatter(max_m_discrete, max_value_discrete, color='red', s=20) # , label=f'Maximum at m={max_m_discrete}')
    plt.legend([r'\textbf{Delay factor}'],  prop={'size': 20})
    plt.xlabel(r'\textbf{Delay profile}', fontsize = 18)
    plt.grid(True, alpha=0.4)
    path = os.path.join(folder, f'zp_{z_p}')
    plt.savefig( path + ".pdf", format = 'pdf', bbox_inches='tight')
    if show:
        plt.show()

    return kappa_list, term1_list, term2_list, term3_list

def plot_profile(n_dots, connections,num_profile,z_p):

    top_dots = [(i, 1) for i in range(n_dots)]
    bottom_dots = [(i, 0) for i in range(n_dots)]


    %matplotlib inline
    fig, ax = plt.subplots(figsize=(25,1))

    for x, y in top_dots + bottom_dots:
        ax.plot(x, y, 'o', color='c', markersize=8)

    for i, j in connections:
        ax.annotate("", xy=bottom_dots[j], xycoords='data', xytext=top_dots[i], textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='c', linewidth=2))

    ax.set_ylim(-0.1, 1.1)

    # folder   = 'check_theorem_figure' 
    folder   = 'new_check_theorem_figure' 

    if not os.path.exists(folder):
        os.makedirs(folder)

    ax.set_aspect('equal')
    plt.axis('off')
    path = os.path.join(folder, f'profile_{num_profile}_zp_{z_p}')
    plt.savefig( path + ".pdf", format = 'pdf', bbox_inches='tight')
    plt.show()

delay_profiles_connections = [[(0,0), (1,1), (2,2), (3,3), (5,5), (6,6), (7,7), (8,8), (9,9)],
                              [(0,0), (1,1), (2,2), (5,5), (6,6), (7,7), (8,8), (9,9)],
                              [(0,0), (1,1), (2,2), (3,3), (4,5), (5,5), (6,6), (7,7), (8,8), (9,9) ],
                              [(0,0), (1,1), (2,2), (3,5), (5,5), (6,6), (7,7), (8,8), (9,9) ],
                              [(0,0), (1,1), (2,2), (3,5), (4,5), (5,5), (6,6), (7,7), (8,8), (9,9) ],
                              [(0,0), (1,1), (2,2), (4,5), (5,5), (6,6), (7,7), (8,8), (9,9) ],
                              [(0,0), (1,1), (2,2), (3,6), (4,5), (5,5), (6,6), (7,7), (8,8), (9,9) ],
                              [(0,0), (1,1), (2,2), (3,3), (4,6), (5,5), (6,6), (7,7), (8,8), (9,9) ],
                              [(0,0), (1,1), (2,2), (3,3), (4,7), (5,5), (6,6), (7,7), (8,8), (9,9) ]
                            #   [(0,0), (1,1), (2,2), (3,8), (4,8), (5,5), (6,6), (7,7), (8,8), (9,9) ]
                              ]

delay_profiles_connections = [[ (2,2), (3,3), (5,5), (6,6), (7,7)],
                              [ (2,2), (5,5), (6,6), (7,7)],
                              [ (2,2), (3,3), (4,5), (5,5), (6,6), (7,7)],
                              [ (2,2), (3,5), (5,5), (6,6), (7,7)],
                              [ (2,2), (3,5), (4,5), (5,5), (6,6), (7,7)],
                              [ (2,2), (4,5), (5,5), (6,6), (7,7)],
                              [ (2,2), (3,6), (4,5), (5,5), (6,6), (7,7)],
                              [ (2,2), (3,3), (4,6), (5,5), (6,6), (7,7)],
                              [ (2,2), (3,3), (4,10), (5,5), (6,6), (7,7)],
                              [ (2,2) , (3,4), (4,3), (5,5), (6,6), (7,7)]
                            #   [(0,0), (1,1), (2,2), (3,8), (4,8), (5,5), (6,6), (7,7), (8,8), (9,9) ]
                              ]

delay_profiles_connections =[[(0, 0), (1, 1), (3, 3), (4, 4), (5, 5)],
                             [(0, 0), (3, 3), (4, 4), (5, 5)],
                             [(0, 0), (1, 1), (2, 3), (3, 3), (4, 4), (5, 5)],
                             [(0, 0), (1, 3), (3, 3), (4, 4), (5, 5)],
                             [(0, 0), (1, 3), (2, 3), (3, 3), (4, 4), (5, 5)],
                             [(0, 0), (2, 3), (3, 3), (4, 4), (5, 5)],
                             [(0, 0), (1, 4), (2, 3), (3, 3), (4, 4), (5, 5)],
                             [(0, 0), (1, 1), (2, 4), (3, 3), (4, 4), (5, 5)]
                            #  [(0, 0), (1, 1), (2, 5), (3, 3), (4, 4), (5,5)],
                            #  [(0, 0), (1, 1), (2, 5), (3, 3), (4, 4), (5,5)],
                            #
                            #  [(0, 0), (1, 1), (2, 3), (3, 2), (4, 4), (5, 5)],
                            #  [(0, 0), (1, 1), (2, 3), (3, 2), (4, 4), (5, 5)],
                            #  [(0, 0), (1, 1), (2, 3), (3, 2), (4, 4), (5, 5)],
                            #  [(0, 0), (1, 1), (2, 3), (3, 2), (4, 4), (5, 5)],
                            #  [(0, 0), (1, 1), (2, 3), (3, 2), (4, 4), (5, 5)]
                             ]

delay_profiles = [
                  [0,0,0,0,1e5,0,0,0,0,0,0],
                  [0,0,0,1e5,1e5,0,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0,0,0,0],
                  [0,0,0,2,1e6,0,0,0,0,0,0],
                  [0,0,0,2,1,0,0,0,0,0,0],
                  [0,0,0,1e7,1,0,0,0,0,0,0],
                  [0,0,0,3,1,0,0,0,0,0,0,0],
                  [0,0,0,0,2,0,0,0,0,0,0,0],
                #   [0,0,0,0,3,0,0,0,0,0,0,0]
                #   [0,0,0,0,6,0,0,0,0,0,0,0,0],
                #
                #   [0,0,0,0,0,0,1,-1,0,0,0,0],
                #   [0,0,0,0,1,-1,1,-1,0,0,0,0],
                #   [0,0,1,-1,1,-1,1,-1,0,0,0,0],
                #   [0,0,3,1,-1,-3,1,-1,0,0,0,0],
                #   [0,0,3,3,3,-3,-3,-3,0,0,0,0]
                  ]

z_p = 0.9
min_K = 0
m_min = -5
m_max = 15
kapaa_list, t1_list, t2_list, t3_list = compute_kappa_and_terms_max(delay_profiles, min_K, z_p, m_min, m_max, show=False )


sorted_indicies_kappa = plot_sorted(kapaa_list, show=False)
reordered_delay_profiles_connections = [delay_profiles_connections[i] for i in sorted_indicies_kappa]
reordered_delay_profiles = [delay_profiles[i] for i in sorted_indicies_kappa]


kappa_final, _, _, _ = compute_kappa_and_terms_max(reordered_delay_profiles, min_K, z_p, m_min, m_max , show=True)



for connections, delays, i in zip(reordered_delay_profiles_connections, reordered_delay_profiles, np.arange(len(delay_profiles_connections))):
    print(f'delay profile {i+1}')
    print(delays)
    plot_profile(6,connections,i+1,z_p)

# for connections, delays, i in zip(delay_profiles_connections, delay_profiles, np.arange(len(delay_profiles_connections))):
#     print(f'delay profile {i}')
#     print(delays)
#     plot_profile(6,connections)
