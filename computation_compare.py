import numpy as np
from ss_problem import SSProblem, SecureStateReconstruct
from tests import generate_random_dtsystem_data, generate_random_io_data,solve_ssr_by_decomposition, remove_duplicate_states
from safe_control import LinearInequalityConstr
from scipy import linalg

import timeit
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Figure 1(a): Runtime comparison between different approaches with varying p
def runtime_compare_varying_p():
    n = 8
    r = n
    m = 8
    # p = 10
    s = 5
    q = 6

    # bf_and_decomp_ssr(n,r,m,p,s,q)

    p_range = range(8,16)
    # p_range = range(8,10)
    seed = 10

    p_list, exe_time_decomp_m, exe_time_decomp, exe_time_bf = [], [], [],[]
    for p in p_range:
        seed = 1
        rng = np.random.default_rng(seed=seed)

        # define system
        dtsys_a, dtsys_b, dtsys_c, dtsys_d = generate_random_dtsystem_data(rng,n,m,p,q,is_jordan=False)

        print(np.shape(dtsys_a), np.shape(dtsys_b), np.shape(dtsys_c), np.shape(dtsys_d))

        # define true and fake initial states
        init_state1 = rng.normal(10,2,(n,1))
        init_state2 = 2*init_state1
        sensor_initial_states = [init_state2 if i < s else init_state1 for i in range(p)]

        # define io data
        u_seq, tilde_y_his, noise_level = generate_random_io_data(dtsys_a,dtsys_b, dtsys_c, dtsys_d, s, sensor_initial_states, 
                                    rng, has_u_seq = True, is_noisy = False)

        # construct a problem instance and a solution instance
        ss_problem = SSProblem(dtsys_a, dtsys_b, dtsys_c, dtsys_d, tilde_y_his, 
                                attack_sensor_count=s,input_sequence=u_seq,measurement_noise_level= noise_level )
        # ssr_solution = SecureStateReconstruct(ss_problem,possible_comb=comb1)
        ssr_solution = SecureStateReconstruct(ss_problem)

        # solve for sparse observability index (soi) and eigenvalue observability index (eoi) 
        soi = SecureStateReconstruct.compute_sparse_observability(ss_problem.A,ss_problem.C)
        eoi = SecureStateReconstruct.compute_eigenvalue_observability(ss_problem.A,ss_problem.C)
        print(f'The problem have a sparse observability index {soi}, eigenvalue observability index: {eoi}, attacked sensor count: {s}')

        # solve ssr by brute-force approach
        possible_states_bruteforce,corresp_sensors, _ = ssr_solution.solve_initial_state(error_bound = 1e-3)
        possible_states_bruteforce = remove_duplicate_states(possible_states_bruteforce.transpose())
        # solve ssr by decomposition
        possible_states_decomposition = solve_ssr_by_decomposition(ssr_solution, eoi,is_composed=True)

        # printing
        print('-----------------------  brute-force   approach  -------------------------------')
        print( possible_states_bruteforce)
            

        print('-----------------------  decomposition approach  -------------------------------')
        print(possible_states_decomposition)
        
        # time it, tqking avarage over 100 computations
        execution_time_decomposition = timeit.timeit(lambda : solve_ssr_by_decomposition(ssr_solution, eoi), number=100)
        execution_time_decomposition_m = timeit.timeit(lambda : solve_ssr_by_decomposition(ssr_solution, eoi,is_composed=False), number=100)

        execution_time_bruteforce = timeit.timeit(lambda : ssr_solution.solve_initial_state(error_bound=1e-3), number=100)
        
        time_decomp = execution_time_decomposition*1000/100
        time_decomp_m = execution_time_decomposition_m*1000/100
        time_bf = execution_time_bruteforce*1000/100

        p_list.append(p)
        exe_time_decomp_m.append(time_decomp_m)
        exe_time_decomp.append(time_decomp)
        exe_time_bf.append(time_bf)

    # Adjust font size and paper size (figure size)
    linestyles = ['-.', '--','-' ]
    # colors = plt.get_cmap('Spectral', len(x_tr_lst))  # Automatically get a colormap
    colors = ['tab:pink','tab:orange','tab:brown','tab:blue']
    labels = ['Brute-force SSR','Decomposition-based SSR','Algorithm 5']

    exe_times = [exe_time_bf,exe_time_decomp,exe_time_decomp_m]

    fig = plt.figure(figsize=(6, 6))
    # plt.figure(figsize=(10, 8))  # Set figure size (width, height in inches)
    plt.rcParams.update({'font.size': 16})  # Globally set the font size

    # plotting
    for i in range(3):
        plt.plot(np.array(p_list), np.array(exe_times[i]),label = labels[i],
                color=colors[i],
                linestyle=linestyles[i],
                linewidth=4)


    # plt.title('Cost Functions for Different Trajectories')
    plt.xlabel('$p$ total number of sensors')
    plt.ylabel(r'Execution time (ms)')
    # plt.ylim([0, 130]) 
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.title(rf'$ n= r = {n}, m = {m}, s = {s}, q = {q}$ ')
    plt.yscale('log')

    # Add a legend
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    # Show the plot
    plt.show()

    # fig.savefig("figures/timing_for_ssr_algorithms_varying_p.pdf")



# Figure 1(b): Runtime comparison between different approaches with varying n
def runtime_compare_varying_n():
    n = 8
    r = n
    m = 8
    p = 12
    s = 5
    q = 6

    # bf_and_decomp_ssr(n,r,m,p,s,q)

    n_range = range(4,10)
    # n_range = range(4,6)
    seed = None


    m_list, exe_time_decomp_m, exe_time_decomp, exe_time_bf = [], [], [],[]
    for n in n_range:
        r = n
        m = n
        seed = 10
        rng = np.random.default_rng(seed=seed)

        # define system
        dtsys_a, dtsys_b, dtsys_c, dtsys_d = generate_random_dtsystem_data(rng,n,m,p,q,is_jordan=False)

        print(np.shape(dtsys_a), np.shape(dtsys_b), np.shape(dtsys_c), np.shape(dtsys_d))

        # define true and fake initial states
        init_state1 = rng.normal(10,2,(n,1))
        init_state2 = 2*init_state1
        sensor_initial_states = [init_state2 if i < s else init_state1 for i in range(p)]

        # define io data
        u_seq, tilde_y_his, noise_level = generate_random_io_data(dtsys_a,dtsys_b, dtsys_c, dtsys_d, s, sensor_initial_states, 
                                    rng, has_u_seq = True, is_noisy = False)

        # construct a problem instance and a solution instance
        ss_problem = SSProblem(dtsys_a, dtsys_b, dtsys_c, dtsys_d, tilde_y_his, 
                                attack_sensor_count=s,input_sequence=u_seq,measurement_noise_level= noise_level )
        # ssr_solution = SecureStateReconstruct(ss_problem,possible_comb=comb1)
        ssr_solution = SecureStateReconstruct(ss_problem)

        # solve for sparse observability index (soi) and eigenvalue observability index (eoi) 
        soi = SecureStateReconstruct.compute_sparse_observability(ss_problem.A,ss_problem.C)
        eoi = SecureStateReconstruct.compute_eigenvalue_observability(ss_problem.A,ss_problem.C)
        print(f'The problem have a sparse observability index {soi}, eigenvalue observability index: {eoi}, attacked sensor count: {s}')

        # solve ssr by brute-force approach
        possible_states_bruteforce,corresp_sensors, _ = ssr_solution.solve_initial_state(error_bound = 1e-3)
        possible_states_bruteforce = remove_duplicate_states(possible_states_bruteforce.transpose())
        # solve ssr by decomposition
        possible_states_decomposition = solve_ssr_by_decomposition(ssr_solution, eoi,is_composed=True)

        # printing
        print('-----------------------  brute-force   approach  -------------------------------')
        print( possible_states_bruteforce)
            

        print('-----------------------  decomposition approach  -------------------------------')
        print(possible_states_decomposition)
        
        # time it, tqking avarage over 100 computations
        execution_time_decomposition = timeit.timeit(lambda : solve_ssr_by_decomposition(ssr_solution, eoi), number=100)
        execution_time_decomposition_m = timeit.timeit(lambda : solve_ssr_by_decomposition(ssr_solution, eoi,is_composed=False), number=100)

        execution_time_bruteforce = timeit.timeit(lambda : ssr_solution.solve_initial_state(error_bound=1e-3), number=100)
        
        time_decomp = execution_time_decomposition*1000/100
        time_decomp_m = execution_time_decomposition_m*1000/100
        time_bf = execution_time_bruteforce*1000/100

        m_list.append(m)
        exe_time_decomp_m.append(time_decomp_m)
        exe_time_decomp.append(time_decomp)
        exe_time_bf.append(time_bf)

    # # plotting
    # fig, ax = plt.subplots()
    # ax.plot(np.array(m_list), np.array(exe_time_bf),label = 'brute-force SSR',color='red',linestyle='-.')
    # ax.plot(np.array(m_list), np.array(exe_time_decomp),label = 'Algorithm 3',color='blue',linestyle='--')
    # ax.plot(np.array(m_list), np.array(exe_time_decomp_m),label = 'Algorithm 5',color='green',linestyle='-')


    # ax.set(xlabel='$n$ dimensional systems', ylabel='Execution time (ms)',
    #     title=rf'$ r = n, m = n, s = {s}, p = {p}$, $ {q}$-eigenvalue observability ')
    # ax.grid()

    # # Add a legend
    # ax.legend()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))



    # Adjust font size and paper size (figure size)
    linestyles = ['-.', '--','-' ]
    # colors = plt.get_cmap('Spectral', len(x_tr_lst))  # Automatically get a colormap
    colors = ['tab:pink','tab:orange','tab:brown','tab:blue']
    labels = ['Brute-force SSR','Decomposition-based SSR','Algorithm 5']

    exe_times = [exe_time_bf,exe_time_decomp,exe_time_decomp_m]

    fig = plt.figure(figsize=(6, 6))
    # plt.figure(figsize=(10, 8))  # Set figure size (width, height in inches)
    plt.rcParams.update({'font.size': 16})  # Globally set the font size

    # plotting
    for i in range(3):
        plt.plot(np.array(m_list), np.array(exe_times[i]),label = labels[i],
                color=colors[i],
                linestyle=linestyles[i],
                linewidth=4)


    # plt.title('Cost Functions for Different Trajectories')
    plt.xlabel('$r$ total number of eigenspaces')
    plt.ylabel(r'Execution time (ms)')
    plt.ylim([0, 130])  # Set y-axis limits
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.title(rf'$  n = m = r, s = {s}, p = {p}, q = {q}$')

    # Add a legend
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    # Show the plot
    plt.show()

    fig.savefig("figures/timing_for_ssr_algorithms_varying_n.pdf")
    


def main():
    runtime_compare_varying_p()
    # runtime_compare_varying_n()

if __name__ =='__main__':
    main()