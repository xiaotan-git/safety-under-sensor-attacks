import numpy as np
from ss_problem import SSProblem, SecureStateReconstruct
from tests import generate_random_dtsystem_data, generate_random_io_data,solve_ssr_by_decomposition, remove_duplicate_states
from safe_control import LinearInequalityConstr
from scipy import linalg
import matplotlib.pyplot as plt
import math

import time

from safe_control import SafeProblem
import random

################# closed-loop system case ####################
############  setting up the simulation 

def solve_safe_control_by_brute_force(ssr_solution:SecureStateReconstruct, safe_prob:SafeProblem, u_nom):
     # solve ssr by brute-force
    possible_states,corresp_sensors, _ = ssr_solution.solve(error_bound = 1e-3)
    possible_states = possible_states.transpose() # now possible_states[0] is one possible state
    possible_states = remove_duplicate_states(possible_states)

    u_safe1,lic1,flag1 = safe_prob.cal_safe_control(u_nom,possible_states)
    if flag1 !=1:
        print('safe_prob.cal_safe_control giving an unsafe input')
    return u_safe1 

def solve_safe_control_by_decomposition(ssr_solution:SecureStateReconstruct, safe_prob:SafeProblem, u_nom, eoi):
    # solve ssr by decomposition
    possible_states_subssr= solve_ssr_by_decomposition(ssr_solution, eoi,is_initial_state=False)
    u_safe2,lic2,flag2 = safe_prob.cal_safe_control(u_nom,possible_states_subssr)

    return u_safe2



def solve_safe_control_woSSR(ssr_solution, safe_prob, u_nom, eoi):
    initial_substates_subssr = solve_ssr_by_decomposition(ssr_solution, eoi, is_print = True,is_composed= False)
    # print(f'initial_substates_subssr: {initial_substates_subssr}')
    lic3 = safe_prob.cal_safe_input_constr_woSSR(initial_substates_subssr)
    u_safe3,flag3 = safe_prob.cal_safe_qp(u_nom,lic3)

    return u_safe3

# system setting

n = 4
r = 4
m = 4

# p >5
p = 11
s = 5
q = 8

# This would take longer for brute-force SSR
p = 20
s = p//2
q = 14


# for reproductivity
seed = 100

rng = np.random.default_rng(seed=seed)
np.random.seed(seed)
random.seed(seed)

##################### Key parameters to experiment #####################
# unstable matrix. row sum = column sum = 1
# dtsys_a = np.array([
#     [ 1.2,  0.3, -0.3, -0.2],
#     [ 0.3,  0.8,  0.1, -0.2],
#     [-0.3,  0.1,  0.9,  0.3],
#     [-0.2, -0.2,  0.3,  1.1]
# ])

# Stable matrix
dtsys_a =np.array([[0.7, 0.3, 0, 0],
                    [0.3, 0.5, 0.2, 0],
                    [0, 0.2, 0.4, 0.2],
                    [0, 0, 0.2, 0.6]])

# unstable matrix
# dtsys_a = np.array([[0.8, 0.4, 0, 0],
#                     [0.4, 0.6, 0.2, 0],
#                     [0, 0.2, 0.5, 0.3],
#                     [0, 0, 0.3, 0.7]])


total_steps = 20
safety_filter_on = False
bf_SSR_control = True
decomp_SSR_control = False # same trajecotry as bf_SSR_control
decomp_M_control = False
gamma = 0.8 
#######################################################################

# dtsys_a = np.random.normal(0,3,(n,n))

dtsys_b = np.identity(n)

# to compute possible C
dtsys_c = np.random.normal(1,1,(p,n))

unique_eig_vals, generalized_space,_,_ = SecureStateReconstruct.compute_sys_eigenproperty(dtsys_a)
# print(f'unique_eig_vals:{unique_eig_vals}')
# print(f'generalized_space: {generalized_space}')
generalized_space = [np.real(space) for space in generalized_space]

unobs_subspace_per_sensor = [[] for _ in range(p)]
for j in range(r):
    # choose p - (q+1) sensors from p sensors for subspace j
    # these sensors do not observe subspace j
    unobs_sensor_subspace_j = np.random.choice(np.arange(0, p), p-(q+1), replace=False)
    for i in range(p):
        if i in unobs_sensor_subspace_j:
            unobs_subspace_per_sensor[i].append(j)

for i in range(p):
    ci = dtsys_c[i:i+1,:]
    ci_tranpose = ci.transpose()
    # each sensor has its own unobservable subspace
    unobs_space_ind = unobs_subspace_per_sensor[i]
    # check if the ubobservable space is empty
    if unobs_space_ind:
        basis_matrix_unobs = np.hstack([ generalized_space[j] for j in  unobs_space_ind])
        bmu_transpose = basis_matrix_unobs.transpose()
        proj_ci_transpose = basis_matrix_unobs @ np.linalg.inv(bmu_transpose @ basis_matrix_unobs) @ bmu_transpose @ ci_tranpose
        ci_remain =  ci_tranpose - proj_ci_transpose
        dtsys_c[i:i+1,:] = ci_remain.transpose()



dtsys_d = np.zeros((p,m))

# generate io_data
# attack random sensors
x_true = np.ones((n,1))
x_fake1 = -np.ones((n,1))
x_fake2 = 2*np.ones((n,1))
sensor_to_inital_state_index= [0]*(p-2-3) + [1]*2 + [2]*3
random.shuffle(sensor_to_inital_state_index)
sensor_initial_states = [ x_true if i == 0 else x_fake1 if i == 1 else x_fake2 for i in sensor_to_inital_state_index]
# sensor_initial_states = [x_true for i in sensor_to_inital_state_index]
print(f'sensor_to_inital_state_index: {sensor_to_inital_state_index}')

u_seq, tilde_y_his, noise_level = generate_random_io_data(dtsys_a,dtsys_b, dtsys_c, dtsys_d, s, sensor_initial_states, 
                            rng, has_u_seq = True, is_noisy = False,io_length=n)



# construct a problem instance and a solution instance
ss_problem = SSProblem(dtsys_a, dtsys_b, dtsys_c, dtsys_d, tilde_y_his, 
                        attack_sensor_count=s,input_sequence=u_seq,measurement_noise_level= noise_level )
# ssr_solution = SecureStateReconstruct(ss_problem,possible_comb=comb1)
ssr_solution = SecureStateReconstruct(ss_problem)

# solve for sparse observability index (soi) and eigenvalue observability index (eoi) 
soi = SecureStateReconstruct.compute_sparse_observability(ss_problem.A,ss_problem.C)
eoi = SecureStateReconstruct.compute_eigenvalue_observability(ss_problem.A,ss_problem.C)
print(f'The problem have a sparse observability index {soi}, eigenvalue observability index: {eoi}, attacked sensor count: {s}')


# define CBF and solve for safe control
h = np.vstack([np.identity(n),-np.identity(n)])
q = 10*np.ones((2*n,1))
u_nom = np.random.normal(0,3,(m,1))

# print(f'u_nom: {u_nom}')
safe_prob = SafeProblem(ssr_solution.problem,h,q,gamma)


## one-step safe control
usafe1 = solve_safe_control_by_brute_force(ssr_solution, safe_prob, u_nom)
usafe2 = solve_safe_control_by_decomposition(ssr_solution, safe_prob, u_nom,eoi)
usafe3 = solve_safe_control_woSSR(ssr_solution, safe_prob, u_nom,eoi)


dt = 1

def get_u_nom(step):
    u_nom = 4*np.array([[math.sin(dt*step)],
                      [math.cos(dt*step)],
                      [-math.sin(dt*step)],
                      [-math.cos(dt*step)]])
    return u_nom

def system_update(A:np.ndarray,B:np.ndarray,C:np.ndarray,D:np.ndarray,x0:np.ndarray,u_seq:np.ndarray,one_step=False):
    # x0: [n,1] or [1,n]
    # u_seq: [duration, m]
    n = A.shape[0]
    p = C.shape[0]
    duration = u_seq.shape[0]

    if not one_step:
        x_seq = np.zeros((n,duration+1))
        x_seq[:,0:1] = x0.reshape(n,1)
        y_seq = np.zeros((duration+1,p))
        y_t = C@x_seq[:,0:1] + D @ np.transpose(u_seq[0:1,:])
        y_seq[0:1,:]  = y_t.transpose()

        for t in range(1,duration+1):
            x_seq[:,t:t+1] = A @ x_seq[:,t-1:t] + B @ np.transpose(u_seq[t-1:t,:])
            y_t = C @ x_seq[:,t-1:t] + D @ np.transpose(u_seq[t-1:t,:])
            y_seq[t:t+1,:] = y_t.transpose()

    if one_step:
        assert duration ==1,f'u_seq.shape = {u_seq.shape}'
        x_seq = A @ x0.reshape(n,1)+ B @ np.transpose(u_seq)
        y_t= C @ x0.reshape(n,1) + D @ np.transpose(u_seq)
        y_seq = y_t.transpose()
    
    # if not one-step: x_seq: [n,duration+1], y_seq: [duration+1,p] 
    # if one_step: x_seq: [n,1], y_seq: [1,p] 
    return x_seq, y_seq


# first several steps
C_true  = np.vstack([dtsys_c[i:i+1,:] for i,x in enumerate(sensor_to_inital_state_index) if x ==0])
D_true = np.zeros((C_true.shape[0],m))

C_fake1 = np.vstack([dtsys_c[i:i+1,:] for i,x in enumerate(sensor_to_inital_state_index) if x ==1])
D_fake1 = np.zeros((C_fake1.shape[0],m))

C_fake2 = np.vstack([dtsys_c[i:i+1,:] for i,x in enumerate(sensor_to_inital_state_index) if x ==2])
D_fake2 = np.zeros((C_fake2.shape[0],m))

# first several steps
# to remove the last row n that are all zeros, u_seq shape (n,m=n)
u_seq_init = u_seq[:-1,:]
# x_first_seq is of size (n,duration+1 = n)
x_first_seq, y_first_seq = system_update(dtsys_a,dtsys_b,C_true,D_true,
                                         x_true,u_seq_init,one_step=False)
x_first_seq_fake1, y_first_seq_fake1 = system_update(dtsys_a,dtsys_b,C_fake1,D_fake1,
                                         x_fake1,u_seq_init,one_step=False)
x_first_seq_fake2, y_first_seq_fake2 = system_update(dtsys_a,dtsys_b,C_fake2,D_fake2,
                                         x_fake2,u_seq_init,one_step=False)

# print(x_first_seq,x_first_seq_fake1,x_first_seq_fake2)

## SSR at the time with n-1 inputs and n measurements
# this should work

# possible_states,corresp_sensors, _ = ssr_solution.solve(error_bound = 1e-3)
# possible_states = possible_states.transpose() # now possible_states[0] is one possible state at time tau
# possible_states = remove_duplicate_states(possible_states)
# for state in possible_states:
#     print('This error should be zero')
#     print(np.linalg.norm(state.reshape((n,1)) - x_first_seq[:,-1:]) )
#     print(f'state:\{state}')
#     print(f'x_first_seq:\n {x_first_seq}')

 



# to keep track of true state, fake state 1, fake state 2
x_tr_mtx = np.zeros((n,n+total_steps+1))
x_fk1_mtx  = np.zeros((n,n+total_steps+1))
x_fk2_mtx  = np.zeros((n,n+total_steps+1))
x_estimate_lst = []

y_mtx = np.zeros((n+total_steps+1,p))
u_seq_mtx = np.zeros((n+total_steps,m))
cost_mtx = np.zeros((2,n+total_steps)) # first row -- two-stage, second row -- one stage Alg.5

## initialization at t = n
x_tr_mtx[:,0:n] = x_first_seq
x_fk1_mtx[:,0:n] = x_first_seq_fake1
x_fk2_mtx[:,0:n] = x_first_seq_fake2


y_mtx[0:n,:] =  tilde_y_his
u_seq_mtx[0:n-1,:] = u_seq_init

# Start timing
start_time = time.time()

###########   discrete-time system simulation   ############
for tau in range(n-1,n+total_steps):
    
    ## User part: preperation and update the SSR_problem
    u_nom = get_u_nom(tau)

    # There is a bug in these two lines
    # ssr_solution.problem.u_seq = u_seq
    # ssr_solution.problem.tilde_y_his = tilde_y_his
    # # we need to update this ssr_solution in as well
    # ssr_solution.y_his = ssr_solution.construct_clean_measurement()

    # construct a problem instance and a solution instance
    ss_problem = SSProblem(dtsys_a, dtsys_b, dtsys_c, dtsys_d, tilde_y_his, 
                            attack_sensor_count=s,input_sequence=u_seq,measurement_noise_level= noise_level )
    # ssr_solution = SecureStateReconstruct(ss_problem,possible_comb=comb1)
    ssr_solution = SecureStateReconstruct(ss_problem)


    # compute the input at t = tau
    # u_tau = solve_safe_control_by_brute_force(ssr_solution, safe_prob, u_nom)
    # possible_states,corresp_sensors, _ = ssr_solution.solve(error_bound = 1e-3)
    # possible_states = possible_states.transpose() # now possible_states[0] is one possible state at time tau
    # possible_states = remove_duplicate_states(possible_states)

    if safety_filter_on:
        # if brute_force_control:
        # u_safe,lic1,flag1 = safe_prob.cal_safe_control(u_nom,possible_states)
        if bf_SSR_control:
            u_safe = solve_safe_control_by_brute_force(ssr_solution,safe_prob,u_nom)
            cost = np.linalg.norm(u_safe.flatten() - u_nom.flatten())

        if decomp_SSR_control:
            u_safe = solve_safe_control_by_decomposition(ssr_solution,safe_prob,u_nom,eoi)
            cost = np.linalg.norm(u_safe.flatten() - u_nom.flatten())

        if decomp_M_control:
            u_safe = solve_safe_control_woSSR(ssr_solution,safe_prob,u_nom,eoi)
            cost = np.linalg.norm(u_safe.flatten() - u_nom.flatten())


        u_tau = u_safe
    else:
        u_tau = u_nom


    u_tau = u_tau.reshape(1,m)
    u_seq_mtx[tau:tau+1,:] = u_tau
    # print(f'possible_states: {possible_states}')
    x_estimate_lst.append(possible_states)

    for state in possible_states:
        print(f'Step {tau}, possible_states - x_tr_now: {np.linalg.norm( state.reshape(n,1) - x_tr_mtx[:,tau:tau+1])}')
    # print(f'CBF-QP cost at step {tau}:{np.linalg.norm(u_nom - u_tau.reshape(m,1))}')

    ## simulating the attacked system. This part is not known to the user
    # x_true_now, x_fake1_now, x_fake2_now
    x_true_now = x_tr_mtx[:,tau:tau+1]
    x_fake1_now = x_fk1_mtx[:,tau:tau+1]
    x_fake2_now = x_fk2_mtx[:,tau:tau+1]


    # compute the true state, fake state 1, fake state 2 at t = tau+1
    x_true_nxt  = dtsys_a@x_true_now + dtsys_b@u_tau.reshape(m,1)
    x_fake1_nxt = dtsys_a@x_fake1_now + dtsys_b@u_tau.reshape(m,1)
    x_fake2_nxt = dtsys_a@x_fake2_now + dtsys_b@u_tau.reshape(m,1)

    # x_true_nxt,_  = system_update(dtsys_a,dtsys_b,C_true,D_true,
    #                                      x_true_now,u_tau,one_step=True)
    # x_fake1_nxt,_ = system_update(dtsys_a,dtsys_b,C_fake1,D_fake1,
    #                                     x_fake1_now,u_tau,one_step=True)
    # x_fake2_nxt,_ = system_update(dtsys_a,dtsys_b,C_fake2,D_fake2,
    #                                     x_fake2_now,u_tau,one_step=True)
    
    # logging x_true_nxt, x_fake1_nxt, x_fake2_nxt
    x_tr_mtx[:,tau+1:tau+2]  = x_true_nxt
    x_fk1_mtx[:,tau+1:tau+2] = x_fake1_nxt 
    x_fk2_mtx[:,tau+1:tau+2] = x_fake2_nxt

    
    # compute the new measurement
    y_nxt_lst =[]
    for i,x_status in enumerate(sensor_to_inital_state_index):
        Ci = dtsys_c[i:i+1,:]
        if x_status == 0:
            y_nxt_lst.append(Ci @ x_true_nxt)
        if x_status == 1:
            y_nxt_lst.append(Ci @ x_fake1_nxt)
        if x_status == 2:
            y_nxt_lst.append(Ci @ x_fake2_nxt)
    y_nxt = np.array(y_nxt_lst).reshape(1,p)

    # x_tr_temp = x_tr_mtx[:,tau-n+2:tau-n+1] # the sequence -n+2, -n+1, ..., 0, 1 has n elements
    # x_fk1_temp = x_fk1_mtx[:,tau-n+2:tau-n+1] # the sequence -n+2, -n+1, ..., 0, 1 has n elements
    # x_fk2_temp = x_fk2_mtx[:,tau-n+2:tau-n+1] # the sequence -n+2, -n+1, ..., 0, 1 has n elements
    # u_seq_temp = u_seq[1:,:] # remove the oldest input
    # u_seq_temp[-1:, :] = u_tau # add the lastest input by replcing the row with all zeros
    # u_seq_temp = np.vstack([u_seq_temp, np.zeros((1,m))]) # add a row of all zeros


    # sensor_initial_states = [ x_tr_temp if i == 0 else x_fk1_temp if i == 1 else x_fk2_temp for i in sensor_to_inital_state_index]
    # u_seq, tilde_y_his, noise_level = generate_random_io_data(dtsys_a,dtsys_b, dtsys_c, dtsys_d, s, sensor_initial_states, 
    #                         rng, has_u_seq = True, u_seq= u_seq_temp, is_noisy = False)

    ## This part is what a user does
    # update input-output data u_seq, tilde_y_his for tau + 1
    # u_seq = u_seq[1:,:] # remove the oldest input
    u_seq[-1:, :] = u_tau # add the lastest input by replcing the row with all zeros
    u_seq = np.vstack([u_seq, np.zeros((1,m))]) # add a row of all zeros

    # tilde_y_his = tilde_y_his[1:,:]
    tilde_y_his = np.vstack([tilde_y_his,y_nxt])

    # only catech 2n input-output history
    if u_seq.shape[0]>2*n:
        u_seq = u_seq[1:,:]
        tilde_y_his = tilde_y_his[1:,:]
    # logging
    y_mtx[tau+1:tau+2,:] = y_nxt

    print(f'u_seq new at step {tau }: {u_seq.shape }')
    print(f'tilde_y_his new at step {tau }: {tilde_y_his.shape }')
    
# End timing
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"closed-loop simulation elapsed time: {elapsed_time:.6f} seconds")

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plt_box(x_min=-2, x_max=2, y_min=-2, y_max=2, color='black', linewidth=2):
    """
    Plots a box boundary on the current figure's axes with specified dimensions and color.

    Parameters:
    x_min, x_max: The x-coordinates of the box boundaries.
    y_min, y_max: The y-coordinates of the box boundaries.
    color: The color of the box boundary.
    linewidth: The width of the box boundary lines.
    """
    # Get the current axis (do not create a new figure)
    ax = plt.gca()

    # Create a rectangle patch for the box
    width = x_max - x_min
    height = y_max - y_min
    rect = patches.Rectangle((x_min, y_min), width, height, 
                             linewidth=linewidth, edgecolor=color, facecolor='none')

    # Add the rectangle to the current axis
    ax.add_patch(rect)

    # Adjust the axis limits to ensure the box is fully visible
    ax.set_xlim(min(ax.get_xlim()[0], x_min), max(ax.get_xlim()[1], x_max))
    ax.set_ylim(min(ax.get_ylim()[0], y_min), max(ax.get_ylim()[1], y_max))

    # Set the aspect ratio to be equal
    ax.set_aspect('equal', adjustable='box')



# Plot the input and trajectory
plt.figure(figsize=(12, 6))

# Plot input signal
plt.subplot(2, 1, 1)
for k in range(m):
    plt.plot(np.arange(n+total_steps), u_seq_mtx[:,k], label=f'Input $u_{k+1}$')
plt.title('Input Signal')
plt.xlabel('Time step')
plt.ylabel('Input')
plt.grid(True)
plt.legend()


# Plot state trajectory
plt.subplot(2, 1, 2)
for i in range(4):
    plt.plot(np.arange(n+total_steps+1), x_tr_mtx[i,:], label=f'State $x_{i+1}$')
plt.title('State Trajectories')
plt.xlabel('Time step')
plt.ylabel('State')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print(f'x_first_seq:\n {x_first_seq}')
print(f'x_estimate_lst:\n {x_estimate_lst}')
print(f'x_tr_mtx:\n {x_tr_mtx[:,0:5]}')

# plot trajectory in 2d
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot( x_tr_mtx[0,:], x_tr_mtx[1,:])
plt.title(f'Trajectory of $x_1, x_2$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True)
# plt.legend()
plt_box(x_min=-10, x_max=10, y_min=-10, y_max=10, color='black', linewidth=2)

plt.subplot(1, 2, 2)
plt.plot( x_tr_mtx[2,:], x_tr_mtx[3,:])
plt.title(f'Trajectory of $x_3, x_4$')
plt.xlabel('$x_3$')
plt.ylabel('$x_4$')
plt.grid(True)
# plt.legend()
plt_box(x_min=-10, x_max=10, y_min=-10, y_max=10, color='black', linewidth=2)
