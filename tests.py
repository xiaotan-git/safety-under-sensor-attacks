import numpy as np
from ss_problem import SSProblem, SecureStateReconstruct
from safe_control import SafeProblem
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


import timeit

def is_same_state(st1:np.ndarray,st2:np.ndarray):
    error = st1.flatten() - st2.flatten()
    # print(f'error norm: {linalg.norm(error)}')
    if linalg.norm(error)<1e-6:
        return True
    return False

def remove_duplicate_states(possible_states):
    state_lst = [possible_states[0]]
    for state in possible_states[1:]:
        if not any(is_same_state(state,st) for st in state_lst):
            state_lst.append(state)
    return state_lst

def compare_two_state_lists(lst1,lst2):
    # check that the possible states from two approaches are the same    
    lst1_contains_lst2 = True
    for st2 in lst2:
        if not any(is_same_state(st1,st2) for st1 in lst1):
            lst1_contains_lst2 = False

    lst2_contains_lst1 = True
    for st1 in lst1:
        if not any(is_same_state(st1,st2) for st2 in lst2):
            lst2_contains_lst1 = False 

    return lst1_contains_lst2, lst2_contains_lst1

def generate_random_dtsystem_data(rng, n:int = 8, m:int = 3, p:int = 5,
                                  is_jordan = False, is_orthgonal = False):

    # dtsys_a = np.random.normal(3,3,(n,n))
    # eig_vals = linalg.eigvals(dtsys_a)
    # this is slow for large matrices
    # while(not all(np.isreal(eig_vals))):
    #     dtsys_a = np.random.normal(3,3,(n,n))
    #     eig_vals = linalg.eigvals(dtsys_a)
    # by construction
    eig_vals = 1.0*rng.random((n,))+0.1 # eig value in [-10,-2] and [2,10]
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
                # print(f'total trials for getting dtsys_A: {trial}')
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

    u_seq, tilde_y_his, noise_level = SSProblem.generate_attack_measurement(dtsys_a, dtsys_b, dtsys_c, dtsys_d,sensor_initial_states,
                                                                            s = s,is_noisy = is_noisy, noise_level=0.0001,has_u_seq = has_u_seq)
    return u_seq,tilde_y_his, noise_level

def init_brute_force_ssr(dtsys_a,dtsys_b, dtsys_c, dtsys_d, s:int,u_seq,tilde_y_his,  noise_level):
    # construct a problem instance
    ss_problem = SSProblem(dtsys_a, dtsys_b, dtsys_c, dtsys_d, tilde_y_his, 
                            attack_sensor_count=s,input_sequence=u_seq,measurement_noise_level= noise_level )
    # ssr_solution = SecureStateReconstruct(ss_problem,possible_comb=comb1)
    ssr_solution = SecureStateReconstruct(ss_problem)
    return ssr_solution

def solve_ssr_by_decomposition(ssr_solution:SecureStateReconstruct, eoi, is_print = True,is_composed= True,is_initial_state = True):
    '''
    when is_composed is True,  returns a list of possible states identified by ssr algorithm
    when is_composed is False, returns a list of possible initial substates in each subspace
    '''
    n = ssr_solution.problem.n
    s = ssr_solution.problem.s
    dtsys_a, dtsys_b, dtsys_c, dtsys_d = ssr_solution.problem.A, ssr_solution.problem.B, ssr_solution.problem.C, ssr_solution.problem.D
    # eigenspace related property
    unique_eigvals, generalized_eigenspace_list, am_list, gm_list = SecureStateReconstruct.compute_sys_eigenproperty(dtsys_a)
    # print(f'Eigen properties. . \n generalized_unique_eigvals

    subprob_a, subprob_c, subprob_y = ssr_solution.generate_subssr_data()
    subspace_states_attacked_sensors_list = []
    initial_substates_subssr = []
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
    
        initial_substates_subssr.append(states)

        states = states.transpose() # to zip
        subspace_states_attacked_sensors = list(zip(states,attacked_sensors_list))
        subspace_states_attacked_sensors_list.append(subspace_states_attacked_sensors) # [..., [xj,[0,1,2]], ...]
    if is_composed:
        full_state_list,corresp_sensors = sub_solution.compose_subspace_states(subspace_states_attacked_sensors_list)
        if full_state_list is None:
            print('--------------composition of subspace state fails---------------')
            for i in range(n):
                print(f'subspace_states_attacked_sensors_list entry{i}: {subspace_states_attacked_sensors_list[i]}')
            return None
        
        if is_initial_state:
            # return all possible initial state
            return full_state_list
        else:
            current_states_list = []
            for init_state in full_state_list:
                init_state = init_state.reshape(-1,1)
                curr_state = ssr_solution.problem.update_state(dtsys_a, dtsys_b, init_state,ssr_solution.problem.u_seq)
                current_states_list.append(curr_state)
                # return all possible current state
            return current_states_list
    
    else:
         # return a list of possible initial substates in each subspace
        return initial_substates_subssr

def initialization(n,m,p,s,seed):
    
    rng = np.random.default_rng(seed=seed)

    # define system
    dtsys_a, dtsys_b, dtsys_c, dtsys_d = generate_random_dtsystem_data(rng,n,m,p)

    # define true and fake initial states
    init_state1 = 10*rng.random((n,1)) -5
    init_state2 = 10*rng.random((n,1)) -5
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

    
    if is_timing:
        execution_time_decomposition = timeit.timeit(lambda : solve_ssr_by_decomposition(ssr_solution, eoi), number=100)
        print(f'execution time by subspace decomposition: {execution_time_decomposition/100 * 1000} ms')

        execution_time_bruteforce = timeit.timeit(lambda : ssr_solution.solve_initial_state(error_bound=1e-3), number=100)
        print(f'execution time by brute-force approach:   {execution_time_bruteforce/100 * 1000} ms')
    
    return possible_states_decomposition,possible_states_bruteforce

def solve_ssr_timing(ssr_solution, eoi):
    execution_time_decomposition = timeit.timeit(lambda : solve_ssr_by_decomposition(ssr_solution, eoi), number=100)

    execution_time_bruteforce = timeit.timeit(lambda : ssr_solution.solve_initial_state(error_bound=1e-3), number=100)
    return execution_time_decomposition/100*1000, execution_time_bruteforce/100*1000


def solve_safe_control_by_brute_force(ssr_solution:SecureStateReconstruct, safe_prob:SafeProblem, u_nom):
     # solve ssr by brute-force
    possible_states,corresp_sensors, _ = ssr_solution.solve(error_bound = 1e-3)
    possible_states = possible_states.transpose() # now possible_states[0] is one possible state
    possible_states = remove_duplicate_states(possible_states)

    u_safe1,lic1,flag1 = safe_prob.cal_safe_control(u_nom,possible_states)

def solve_safe_control_by_decomposition(ssr_solution:SecureStateReconstruct, safe_prob:SafeProblem, u_nom, eoi):
    # solve ssr by decomposition
    possible_states_subssr= solve_ssr_by_decomposition(ssr_solution, eoi,is_initial_state=False)
    u_safe2,lic2,flag2 = safe_prob.cal_safe_control(u_nom,possible_states_subssr)



def solve_safe_control_woSSR(ssr_solution, safe_prob, u_nom, eoi):
    initial_substates_subssr = solve_ssr_by_decomposition(ssr_solution, eoi, is_print = True,is_composed= False)
    # print(f'initial_substates_subssr: {initial_substates_subssr}')
    lic3 = safe_prob.cal_safe_input_constr_woSSR(initial_substates_subssr)
    u_safe3,flag3 = safe_prob.cal_safe_qp(u_nom,lic3)

def solve_safe_control_timing(ssr_solution, safe_prob, u_nom, eoi):
    
    execution_time_bruteforce = timeit.timeit(lambda : solve_safe_control_by_brute_force(ssr_solution, safe_prob, u_nom), number=100)
    execution_time_decomposition = timeit.timeit(lambda : solve_safe_control_by_decomposition(ssr_solution, safe_prob, u_nom, eoi), number=100)
    execution_time_woSSR = timeit.timeit(lambda : solve_safe_control_woSSR(ssr_solution, safe_prob, u_nom, eoi), number=100)
    
    return execution_time_decomposition/100*1000, execution_time_bruteforce/100*1000, execution_time_woSSR/100*1000

def main_ssr():
    n = 8
    m = 3
    p = 8
    s = 4
    seed = None

    ssr_solution, eoi, init_state1, init_state2 = initialization(n,m,p,s,seed)
    possible_states_decomposition,possible_states_bruteforce = solve_ssr(ssr_solution, eoi, init_state1, init_state2)
    possible_states_bruteforce = possible_states_bruteforce.transpose() # so that possible_states_bruteforce[0] is a state
    flag1,flag2 = compare_two_state_lists(possible_states_bruteforce,possible_states_decomposition)
    assert (flag1 and flag2)

    # printing
    for full_state in possible_states_decomposition:
        print('-----------------------  decomposition approach  -------------------------------')
        if np.linalg.norm(full_state.reshape(-1,1)-init_state1)<1e-3:
            print(f'Initial state 1 is a plausible state based on SSR by decomposition approach')
        if np.linalg.norm(full_state.reshape(-1,1)-init_state2)<1e-3:
            print(f'Initial state 2 is a plausible state based on SSR by decomposition approach')

    for full_state in possible_states_bruteforce:
        print('-----------------------  brute-force   approach  -------------------------------')
        if np.linalg.norm(full_state.reshape(-1,1)-init_state1)<1e-3:
            print(f'Initial state 1 is a plausible state based on SSR by brute-force approach')
        if np.linalg.norm(full_state.reshape(-1,1)-init_state2)<1e-3:
            print(f'Initial state 2 is a plausible state based on SSR by brute-force approach')

def main_ssr_timing(n = 8, m = 3, s = 4, p_range=range(8,18)):
  
    seed = None

    p_list, exe_time_decomp, exe_time_bf = [], [], []
    for p in p_range:
        ssr_solution, eoi, init_state1, init_state2 = initialization(n,m,p,s,seed)
        time_decomp, time_bf = solve_ssr_timing(ssr_solution, eoi)
        p_list.append(p)
        exe_time_decomp.append(time_decomp)
        exe_time_bf.append(time_bf)
    
    # plotting
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

def main_secure_and_safe_control():
    n = 4
    m = 4
    p = 7
    s = 4
    seed = None

    print(f'S&S problem has {n} states, {m} inputs, {p} sensors, {s} attacked sensors.')
    # initialization
    ssr_solution, eoi, init_state1, init_state2 = initialization(n,m,p,s,seed)

    # solve ssr by brute-force
    possible_states,corresp_sensors, _ = ssr_solution.solve(error_bound = 1e-3)
    possible_states = possible_states.transpose() # now possible_states[0] is one possible state
    possible_states = remove_duplicate_states(possible_states)

    # solve ssr by decomposition
    possible_states_subssr= solve_ssr_by_decomposition(ssr_solution, eoi,is_initial_state=False)


    # define CBF and solve safe control
    h = np.vstack([np.identity(n),-np.identity(n)])
    q = 10*np.ones((2*n,1))
    gamma = 0.8
    u_nom = np.random.normal(0,3,(m,1))
    # print(f'u_nom: {u_nom}')
    safe_prob = SafeProblem(ssr_solution.problem,h,q,gamma)

    print('\n -----------------------  brute-force   approach  -------------------------------')
    u_safe1,lic1,flag1 = safe_prob.cal_safe_control(u_nom,possible_states)
    print(f'possible_states:{possible_states}')
    print(f'u_safe: {u_safe1}')
    print(f'||u_safe - u_nom||:{linalg.norm(u_nom -u_safe1.reshape(-1,1) )}')

    print('\n -----------------------  decomposition approach  -------------------------------')
    u_safe2,lic2,flag2 = safe_prob.cal_safe_control(u_nom,possible_states_subssr)
    print(f'possible_states:{[state.flatten() for state in possible_states_subssr]}')
    print(f'u_safe: {u_safe2}')
    print(f'||u_safe - u_nom||:{linalg.norm(u_nom -u_safe2.reshape(-1,1) )}')

    print('\n -----------------------  decomposition without combination -----------------------')
    # safe control with ssr by decomposition and not composed back
    initial_substates_subssr = solve_ssr_by_decomposition(ssr_solution, eoi, is_print = True,is_composed= False)
    # print(f'initial_substates_subssr: {initial_substates_subssr}')
    lic3 = safe_prob.cal_safe_input_constr_woSSR(initial_substates_subssr)
    u_safe3,flag3 = safe_prob.cal_safe_qp(u_nom,lic3)
    print(f'u_safe: {u_safe3}')
    print(f'||u_safe - u_nom||:{linalg.norm(u_nom -u_safe3.reshape(-1,1) )}')


    # if the safe qp is solved accurately, then all these should be True 
    print(f'\n --------------- Assertions--------------')
    print(f'lic1.is_satisfy(u_safe1): {lic1.is_satisfy(u_safe1)}')
    print(f'lic2.is_satisfy(u_safe2): {lic2.is_satisfy(u_safe2)}')
    print(f'lic3.is_satisfy(u_safe3): {lic3.is_satisfy(u_safe3)}')

    print(f'lic1.is_satisfy(u_safe3): {lic1.is_satisfy(u_safe3)}')
    print(f'lic2.is_satisfy(u_safe3): {lic2.is_satisfy(u_safe3)}')
    print(f'control derivation norm difference 1 and 3: {linalg.norm(u_safe1.reshape(-1,1) - u_nom) <= 1e-4+ linalg.norm(u_safe3.reshape(-1,1) - u_nom)}')
    print(f'control derivation norm difference 2 and 3: {linalg.norm(u_safe2.reshape(-1,1) - u_nom) <= 1e-4+ linalg.norm(u_safe3.reshape(-1,1) - u_nom)}')

def main_secure_and_safe_control_timing(n = 8, m = 3, s = 4, p_range=range(8,18)):
    seed = None

    p_list, exe_time_decomp, exe_time_bf, exe_time_woSSR= [], [], [],[]
    for p in p_range:
        ssr_solution, eoi, init_state1, init_state2 = initialization(n,m,p,s,seed)
        
        # define CBF and solve safe control
        h = np.vstack([np.identity(n),-np.identity(n)])
        q = 10*np.ones((2*n,1))
        gamma = 0.8
        u_nom = np.random.normal(0,3,(m,1))
        # print(f'u_nom: {u_nom}')
        safe_prob = SafeProblem(ssr_solution.problem,h,q,gamma)

        time_decomp, time_bf, time_woSSR = solve_safe_control_timing(ssr_solution, safe_prob, u_nom, eoi)
        p_list.append(p)
        exe_time_decomp.append(time_decomp)
        exe_time_bf.append(time_bf)
        exe_time_woSSR.append(time_woSSR)
    
    # plotting
    fig, ax = plt.subplots()
    ax.plot(np.array(p_list), np.array(exe_time_decomp),label = 'decomposition',color='blue')
    ax.plot(np.array(p_list), np.array(exe_time_bf),label = 'brute-force',color='red')
    ax.plot(np.array(p_list), np.array(exe_time_woSSR),label = 'decomposition without combination',color='orange')

    ax.set(xlabel='$p$ total number of sensors', ylabel='Safe control computation time (ms)',
        title=rf'$n = {n}, m  = {m}, s = {s}$, eigenvalue sparse observability $= p-1$')
    ax.grid()

    # Add a legend
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    fig.savefig("figures/timing_for_SS_algorithms.pdf")
    # Show the plot
    plt.show()

if __name__=='__main__':
    # main_ssr()
    # main_ssr_timing()
    main_secure_and_safe_control()

    