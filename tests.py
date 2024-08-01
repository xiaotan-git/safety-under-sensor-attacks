import numpy as np
from ss_problem import SSProblem, SecureStateReconstruct
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import timeit


def generate_random_dtsystem_data(rng, n:int = 8, m:int = 3, p:int = 5,
                                  is_jordan = False, is_orthgonal = False):

    # dtsys_a = np.random.normal(3,3,(n,n))
    # eig_vals = linalg.eigvals(dtsys_a)
    # this is slow for large matrices
    # while(not all(np.isreal(eig_vals))):
    #     dtsys_a = np.random.normal(3,3,(n,n))
    #     eig_vals = linalg.eigvals(dtsys_a)
    # by construction
    eig_vals = 8*rng.random((n,))+2 # eig value in [-10,-2] and [2,10]
    eig_vals = np.multiply(np.sign(rng.random((n,)) - 1.0/2),eig_vals)
    diag_eig = np.diag(eig_vals)

    if not is_orthgonal:
        trial = 0
        while True: 
            P = rng.random((n, n))
            for i in range(n):
                pi = P[:,i:i+1]
                pi_nom = pi / (np.linalg.norm(pi) + 1e-10)
                P[:,i:i+1] = pi_nom
            try:
                trial = trial + 1
                Pinv = np.linalg.pinv(P)
                P_det = np.linalg.det(P)
                if P_det < 0.01:
                    raise np.linalg.LinAlgError
                print(f'total trials for getting dtsys_A: {trial}')
                break
            except np.linalg.LinAlgError:
                pass

    if is_orthgonal:
    # choose an orthogonal basis
        Q = rng.random((n, n))
        P,_ = linalg.qr(Q)
        Pinv = P.transpose()

    dtsys_a = P@diag_eig@Pinv
    # print(f'Ac eig_vals: {eig_vals}')
    # print(f'dtsys_a: {dtsys_a}')
    assert all(np.isreal(eig_vals))

    dtsys_b = np.random.normal(1,1,(n,m))
    dtsys_c = np.random.normal(1,1,(p,n)) # if given one sensor redundancy
    dtsys_d = np.zeros((p,m))
    return dtsys_a,dtsys_b, dtsys_c, dtsys_d

def generate_random_io_data(dtsys_a,dtsys_b, dtsys_c, dtsys_d, s:int, sensor_initial_states, 
                            rng, has_u_seq = False, is_noisy = False):
    n = dtsys_a.shape[1]
    p = dtsys_c.shape[0]

    if has_u_seq:
        m = dtsys_b.shape[1]
        u_seq = rng.random((n,m))
    else:
        u_seq = None

    u_seq, tilde_y_his, noise_level = SSProblem.generate_attack_measurement(dtsys_a, dtsys_b, dtsys_c, dtsys_d,sensor_initial_states,
                                                                            s = s,is_noisy = is_noisy, noise_level=0.0001,u_seq = u_seq)
    return u_seq,tilde_y_his, noise_level

def init_brute_force_ssr(dtsys_a,dtsys_b, dtsys_c, dtsys_d, s:int,u_seq,tilde_y_his,  noise_level):
    # construct a problem instance
    ss_problem = SSProblem(dtsys_a, dtsys_b, dtsys_c, dtsys_d, tilde_y_his, 
                            attack_sensor_count=s,input_sequence=u_seq,measurement_noise_level= noise_level )
    # ssr_solution = SecureStateReconstruct(ss_problem,possible_comb=comb1)
    ssr_solution = SecureStateReconstruct(ss_problem)
    return ssr_solution

# decompose ssr to sub_ssr problems
def solve_ssr_by_decomposition(ssr_solution, eoi, is_print = True):
    n = ssr_solution.problem.n
    s = ssr_solution.problem.s
    dtsys_a, dtsys_b, dtsys_c, dtsys_d = ssr_solution.problem.A, ssr_solution.problem.B, ssr_solution.problem.C, ssr_solution.problem.D
    # eigenspace related property
    unique_eigvals, generalized_eigenspace_list, am_list, gm_list = SecureStateReconstruct.compute_sys_eigenproperty(dtsys_a)
    # print(f'Eigen properties. . \n generalized_unique_eigvals

    subprob_a, subprob_c, subprob_y = ssr_solution.generate_subssr_data()
    subspace_states_attacked_sensors_list = []
    for j in range(subprob_a.shape[2]):
        sub_a = subprob_a[:,:,j]
        sub_c = subprob_c
        sub_y_his = subprob_y[:,:,j]

        subproblem = SSProblem(sub_a, dtsys_b, sub_c, dtsys_d,sub_y_his, 
                                attack_sensor_count=s,is_sub_ssr=True)
        sub_solution = SecureStateReconstruct(subproblem)
        states, attacked_sensors_list = sub_solution.solve_initial_state_subssr(
                        eoi, generalized_eigenspace_list[j],error_bound = 1
            )

        if states is None:
            print(f'-------------solving subspace state x{j} fails---------------')
            return None
    
        states = states.transpose() # to zip
        subspace_states_attacked_sensors = list(zip(states,attacked_sensors_list))
        subspace_states_attacked_sensors_list.append(subspace_states_attacked_sensors) # [..., [xj,[0,1,2]], ...]

    full_state_list,corresp_sensors = sub_solution.compose_subspace_states(subspace_states_attacked_sensors_list)
    if full_state_list is None:
        print('--------------composition of subspace state fails---------------')
        for i in range(n):
            print(f'subspace_states_attacked_sensors_list entry{i}: {subspace_states_attacked_sensors_list[i]}')
        return None
    
    return full_state_list

def initialization(n,m,p,s,seed):
    
    rng = np.random.default_rng(seed=seed)

    # define system
    dtsys_a, dtsys_b, dtsys_c, dtsys_d = generate_random_dtsystem_data(rng,n,m,p)

    # define true and fake initial states
    init_state1 = rng.normal(10,2,(n,1))
    init_state2 = 2*init_state1
    sensor_initial_states = [init_state2 if i < s else init_state1 for i in range(p)]

    # define io data
    u_seq, tilde_y_his, noise_level = generate_random_io_data(dtsys_a,dtsys_b, dtsys_c, dtsys_d, s, sensor_initial_states, 
                                rng, has_u_seq = False, is_noisy = False)

    # construct a problem instance and a solution instance
    ss_problem = SSProblem(dtsys_a, dtsys_b, dtsys_c, dtsys_d, tilde_y_his, 
                            attack_sensor_count=s,input_sequence=u_seq,measurement_noise_level= noise_level )
    # ssr_solution = SecureStateReconstruct(ss_problem,possible_comb=comb1)
    ssr_solution = SecureStateReconstruct(ss_problem)

    # solve for sparse observability index (soi) and eigenvalue observability index (eoi) 
    soi = SecureStateReconstruct.compute_sparse_observability(ss_problem.A,ss_problem.C)
    eoi = SecureStateReconstruct.compute_eigenvalue_observability(ss_problem.A,ss_problem.C)
    print(f'The problem have a sparse observability index {soi}, eigenvalue observability index: {eoi}, attacked sensor count: {s}')
    return ssr_solution, eoi, init_state1, init_state2

def solve_ssr(ssr_solution:SecureStateReconstruct, eoi:int, init_state1, init_state2,is_timing=True):
    # solve ssr by decomposition
    possible_states_decomposition = solve_ssr_by_decomposition(ssr_solution, eoi)

    # solve ssr by brute-force
    possible_states_bruteforce,corresp_sensors, _ = ssr_solution.solve_initial_state(error_bound = 1e-3)

    # printing
    for full_state in possible_states_decomposition:
        print('-----------------------  decomposition approach  -------------------------------')
        if np.linalg.norm(full_state.reshape(-1,1)-init_state1)<1e-3:
            print(f'Initial state 1 is a plausible state based on SSR by decomposition approach')
        if np.linalg.norm(full_state.reshape(-1,1)-init_state2)<1e-3:
            print(f'Initial state 2 is a plausible state based on SSR by decomposition approach')

    for full_state in possible_states_bruteforce.transpose():
        print('-----------------------  brute-force   approach  -------------------------------')
        if np.linalg.norm(full_state.reshape(-1,1)-init_state1)<1e-3:
            print(f'Initial state 1 is a plausible state based on SSR by brute-force approach')
        if np.linalg.norm(full_state.reshape(-1,1)-init_state2)<1e-3:
            print(f'Initial state 2 is a plausible state based on SSR by brute-force approach')
    
    if is_timing:
        execution_time_decomposition = timeit.timeit(lambda : solve_ssr_by_decomposition(ssr_solution, eoi), number=100)
        print(f'execution time by subspace decomposition: {execution_time_decomposition/100 * 1000} ms')

        execution_time_bruteforce = timeit.timeit(lambda : ssr_solution.solve_initial_state(error_bound=1e-3), number=100)
        print(f'execution time by brute-force approach:   {execution_time_bruteforce/100 * 1000} ms')

def solve_ssr_timing(ssr_solution, eoi):
    execution_time_decomposition = timeit.timeit(lambda : solve_ssr_by_decomposition(ssr_solution, eoi), number=100)

    execution_time_bruteforce = timeit.timeit(lambda : ssr_solution.solve_initial_state(error_bound=1e-3), number=100)
    return execution_time_decomposition/100*1000, execution_time_bruteforce/100*1000

def main():
    n = 8
    m = 3
    p = 8
    s = 4
    seed = None

    ssr_solution, eoi, init_state1, init_state2 = initialization(n,m,p,s,seed)
    solve_ssr(ssr_solution, eoi, init_state1, init_state2)

def main_test_timing(n = 8, m = 3, s = 4, p_range=range(9,18)):
  
    seed = None

    p_list, exe_time_decomp, exe_time_bf = [], [], []
    for p in p_range:
        ssr_solution, eoi, init_state1, init_state2 = initialization(n,m,p,s,seed)
        time_decomp, time_bf = solve_ssr_timing(ssr_solution, eoi)
        p_list.append(p)
        exe_time_decomp.append(time_decomp)
        exe_time_bf.append(time_bf)
    
    fig, ax = plt.subplots()
    ax.plot(np.array(p_list), np.array(exe_time_decomp),label = 'decomposition',color='blue')
    ax.plot(np.array(p_list), np.array(exe_time_bf),label = 'brute-force',color='red')

    ax.set(xlabel='$p$ total number of sensors', ylabel='SSR execution time (ms)',
        title=rf'$n = {n}, m  = {m}, s = {s}$, eigenvalue sparse observability $= p-1$')
    ax.grid()

    # Add a legend
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    fig.savefig("figures/timing_for_ssr_algorithms.pdf")
    # Show the plot
    plt.show()



if __name__=='__main__':
    # main()
    main_test_timing()