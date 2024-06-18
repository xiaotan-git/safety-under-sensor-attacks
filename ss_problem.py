from typing import List
import itertools
import numpy as np
import control as ct

# sampling rate
TS = 0.04

def nchoosek(v: List[int], k: int) -> List[List[int]]:
    """
    Returns a list of lists containing all possible combinations of the elements of vector v taken k at a time.
    
    Args:
        v (List[int]): A list of elements to take combinations from.
        k (int): The number of elements in each combination.
    
    Returns:
        List[List[int]]: A list of lists where each sublist is a combination of k elements from v.
    """
    return [list(comb) for comb in itertools.combinations(v, k)]

def right_shift_row_array(a, shift_amount):
    '''
    shift a row array (1,d) rightwards by shift_amount
    a = np.array([a1, a2, a3]) or a = np.array([[a1, a2, a3]]) or a = np.array([[a1], [a2], [a3]])git
    a = right_shift_row_array(a,2) 
    # result 
    a = np.array([[a3,a2,a1]])
    '''
    a.reshape(1,-1)
    to_shift = a[0,-shift_amount:]
    a_shift = np.concatenate((to_shift,a[0,0:-shift_amount]))
    a_shift.reshape(1,-1)
    return a_shift

class SSProblem():
    '''
    This class defines data, system model, etc for SSR problem
    '''
    def __init__(self, Ac, Bc, Cc, Dc,s=2) -> None:
        A,B,C,D = self.ct_to_dt(Ac, Bc, Cc, Dc, TS)

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.n = np.shape(Ac)[0] # dim of states
        self.m = np.shape(Bc)[1] # dim of inputs
        self.p = np.shape(Cc)[0] # no. of sensors
        self.s = s # max no. of attacked sensors

        assert self.n == np.shape(self.A)[0]
        assert self.n == np.shape(self.A)[1]
        assert self.n == np.shape(self.B)[0]
        assert self.m == np.shape(self.B)[1]
        assert self.p == np.shape(self.C)[0]
        assert self.n == np.shape(self.C)[1]

    def ct_to_dt(self,Ac, Bc, Cc, Dc, ts):
        '''
        From continuous time system (Ac, Bc, Cc, Dc) to discrete-time system (A,B,C,D) with ZOH discretization scheme 
        '''
        sys = ct.ss(Ac, Bc, Cc, Dc)
        dtsys = sys.sample(ts,method = 'zoh')
        return dtsys.A, dtsys.B, dtsys.C, dtsys.D, 


    def update_state_one_step(self,xt,ut):
        '''
        This method updates an array of system states xt given an input ut

        '''
        # xt is of dimension n \times no. of possible states
        xt.reshape(self.n,-1)
        ut.reshape(self.m,1)
        x_new = self.A@xt + (self.B@ut).reshape(self.n,1) # broadcasting
        return x_new
    
    def update_state(self,xt,u_seq):
        '''
        This method updates an array of system states xt given an input sequence u_seq. 
        x: (n,), (n,1), (1,n) array
        u_seq: (t,m) array for multiple steps, (m,), (m,1), (1,m) array for one step update
        '''
        if u_seq.size == self.m:
            x_new = self.update_state_one_step(xt,u_seq)
        else:
            x_old = xt
            for t in range(u_seq.shape[0]):
                x_new = self.update_state_one_step(x_old,u_seq[t,:])
                x_old = x_new
        return x_new
    
    def state_fr_sensor(self,sensor_ind,states):
        '''
        This method maps a sensor index to the corresp. state in an array of states acoording to att_dic
        '''
        att_dic = self.att_dic
        att_lst = list(att_dic.keys())
        corres_state_ind = [a for a in att_lst if sensor_ind in att_dic.get(a)]
        assert len(corres_state_ind) == 1
        corres_state_ind = corres_state_ind[0]
        corres_state = states[:,corres_state_ind]
        return corres_state
        
    def gen_attack_measurement(self,tspan = None,s = None,att_dic = None,
                                fake_state_count = 1, init_states = None,u_seq = None,
                                noise = False,noise_level = 0.001):
        '''
        Generate the worst-case attack by assuming that the attacker tries to confuse the system with some states. 
        Which sensor to confuse the system with which state is given in att_dic
        '''
        # no. of measurement steps
        if tspan is None:
            tspan = self.n
        # no. of sensors attacked
        if s is None:
            s = self.s
        
        # initial states (the first column is true, other columns are fake)
        if init_states is None:
            # randomize init_state
            init_states = np.random.uniform(-5,5,(self.n, fake_state_count+1))
        # input sequence
        if u_seq is None:
        # random input_sequence, oldest input comes first, e.g., u_seq[0,:] -> u(0) 
            u_seq = np.random.uniform(-5,5,(tspan,self.m))
            u_seq[-1,:] = 0.0

        assert fake_state_count<= s
        assert init_states.shape[1] == len(att_dic)

        self.tspan = tspan
        self.s = s
        self.init_states = init_states
        self.att_dic = att_dic
        self.u_seq = u_seq
        self.noise_level = noise_level

        measurement_y = np.zeros((tspan,self.p))
        xt_new = init_states
        for t in range(tspan):
            for i in range(self.p):
                corres_state = self.state_fr_sensor(i,xt_new)
                # print(f'sensor {i}, corres. state {corres_state}')
                Ci = self.C[i:i+1,:] # 2D array
                Di = self.D[i:i+1,:] # 2D array
                yi_t = Ci@corres_state + Di@(u_seq[t,:].reshape(self.m,1))
                measurement_y[t,i] = yi_t
            # update state to next time instant
            xt_old = xt_new
            xt_new = self.update_state(xt_old,u_seq[t,:])

        if noise:
            measurement_y = measurement_y + np.random.normal(0,noise_level,(tspan,self.p))

        self.tilde_y_his = measurement_y

class SecureStateReconst():
    '''
    This class implements different SSR algorithms and yields possible states and corresponding sensors
    '''
    def __init__(self, ss_problem:SSProblem,possible_comb = None) -> None:
        self.problem = ss_problem
        obser_array = np.zeros((ss_problem.tspan,ss_problem.n,ss_problem.p))

        A = ss_problem.A
        for i in range(ss_problem.p):
            Ci = ss_problem.C[i:i+1,:]
            obser_i = Ci@np.linalg.matrix_power(A,0)
            for t in range(1,ss_problem.tspan):
                new_row = Ci@np.linalg.matrix_power(A,t)
                obser_i = np.vstack((obser_i,new_row))
            obser_array[:,:,i:i+1] = obser_i.reshape(ss_problem.tspan,ss_problem.n,1)
        self.obser = obser_array

        # possible healthy sensor combinations
        if possible_comb is None:
            num_healthy_sensors = ss_problem.p-ss_problem.s
            self.possible_comb = nchoosek([i for i in range(ss_problem.p)],num_healthy_sensors)

        tilde_y = ss_problem.tilde_y_his
        u_seq = ss_problem.u_seq
        tspan = ss_problem.tspan
        u_list = [u_seq[t,:] for t in range(tspan)]
        u_vec = np.vstack(u_list).reshape(-1,1) # 2d column array

        # yi
        yi_list = []
        for i in range(ss_problem.p):
            tilde_yi = tilde_y[:,i:i+1] #2d array of dimension (tspan,1)
            fi = self.construct_fi(i) #2d array of dimension (tspan,tspan*m)
            yi = tilde_yi - fi@u_vec #2d array of dimension (tspan,1)
            yi_list.append(yi)
        
        self.y_his = np.hstack(yi_list)

    def construct_fi(self,i):
        '''
        construct Fi according to Eq. 5 in the paper
        '''
        Ci = self.problem.C[i,:]
        A = self.problem.A
        B = self.problem.B
        tspan = self.problem.tspan
        m = self.problem.m
       
        fi = np.zeros((tspan,tspan*m))
        for t in range(1,tspan):
            fi[t:t+1,:] = right_shift_row_array(fi[t-1:t,:],m)
            # here at the t-th row, t = 1,...,tspan -1, the left most element is Ci A^t B. 
            fi[t:t+1,0:m] = Ci @ np.linalg.matrix_power(A,t) @ B 
        return fi

    def construct_obser_matrix(self,comb):
        obser_matrix_list = []
        for i in comb:
            obser_i = self.obser[:,:,i]
            obser_matrix_list.append(obser_i)
        obser_matrix = np.vstack(obser_matrix_list)
        return obser_matrix
    
    def construct_measurement_his(self,comb):
        measure_vec_list = []
        for i in comb:
            measure_i = self.y_his[:,i:i+1]
            measure_vec_list.append(measure_i)
        measure_vec = np.vstack(measure_vec_list)
        return measure_vec

    def solve_initial_state(self,error_bound = 1):
        '''
        The method solves a given SSR problem and yields possible initial states, currently in a brute-force approach. 
        '''
        possible_states_list = []
        corresp_sensors_list = []
        residuals_list = []
        for comb in self.possible_comb:
            obser_matrix = self.construct_obser_matrix(comb)
            measure_vec = self.construct_measurement_his(comb)

            # print(f'Observation matrix shape {obser_matrix.shape}, measure_vec shape {measure_vec.shape}')
            state, residuals, rank, _ = np.linalg.lstsq(obser_matrix,measure_vec,rcond=-1)

            if len(residuals)<1:
                # print(f'combinations: {comb}')
                residuals = np.linalg.norm(obser_matrix@state - measure_vec,ord=2)**2

            if residuals <error_bound:
                possible_states_list.append(state)
                corresp_sensors_list.append(comb)
                residuals_list.append(residuals)
                # print(f'residuals: {residuals}')
                if rank < obser_matrix.shape[1]:
                    print(f'Warning: observation matrix for sensors in {comb} is of deficient rank {rank}.')

        if len(possible_states_list)>0:
            # here we remove the states that yields 100x smallest residual
            residual_min = min(residuals_list)
            comb_list = [i for i in range(len(residuals_list)) if residuals_list[i]<10*residual_min]
            possible_states_list = [possible_states_list[index] for index in comb_list]
            corresp_sensors_list = [corresp_sensors_list[index] for index in comb_list]
            possible_states = np.hstack(possible_states_list)
            corresp_sensors = np.array(corresp_sensors_list)
        else:
            possible_states = None
            corresp_sensors = None
            print('No possible state found. Consider relax the error bound')

        return possible_states, corresp_sensors, corresp_sensors_list
    
    def solve(self,error_bound = 1):
        possible_states, corresp_sensors, corresp_sensors_list = self.solve_initial_state(error_bound)
        current_states_list = []
        for ind in range(possible_states.shape[1]):
            init_state = possible_states[:,ind]
            curr_state = self.problem.update_state(init_state,self.problem.u_seq)
            current_states_list.append(curr_state)
        current_states = np.hstack(current_states_list)
        return current_states, corresp_sensors, corresp_sensors_list

if __name__ == "__main__":
    # define system model and measurement model
    Ac = np.array([[0, 1, 0, 0],[0, -0.2, 0, 0],[0,0,0,1],[0,0,0,-0.2]])
    Bc = np.array([[0, 0],[1, 0],[0, 0],[0,1]])
    Cc = np.array([[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]])
    # Cc = np.vstack((Cc,Cc)) # if given enough sensor redundancy
    Dc = np.zeros((Cc.shape[0],Bc.shape[1]))
    p = np.shape(Cc)[0]
    s = 3
    ss_problem = SSProblem(Ac,Bc,Cc,Dc,s)

    # define attacking model
    fake_state_count = 1
    init_states = np.array([[1.,2.],[1.,2.],[1.,2.],[1.,1.]])
    att_ind = [0,2,4] # sensors 1, 3, 5
    # att_dic: which initial state and its corresponding sensors
    # 0 -- healthy sensor, 1, 2, ... -- attacked sensors
    att_dic = {0:[i for i in range(p) if i not in att_ind], 1:att_ind}

    ss_problem.gen_attack_measurement(s=3,att_dic = att_dic,fake_state_count=fake_state_count,
                                        init_states=init_states,noise=True)
    print(f'A: {ss_problem.A},  \n B: {ss_problem.B}, \n C:{ss_problem.C}, \n D:{ss_problem.D}')
    print('input_sequence:',ss_problem.u_seq)
    print('measurement_y:',ss_problem.tilde_y_his)
    print('(To be determined) true tates:',ss_problem.init_states[:,0])
    print('(To be determined) fake tates:',ss_problem.init_states[:,1])
    print(f'(To be determined) Attacked sensor(s) {ss_problem.att_dic.get(1)}')

    # define a solution instance
    ssr_solution = SecureStateReconst(ss_problem)
    possible_states,corresp_sensors, _ = ssr_solution.solve(error_bound = 1e-2)
    if possible_states is not None:
        for ind in range(corresp_sensors.shape[0]):
            sensors = corresp_sensors[ind,:]
            state = possible_states[:,ind]
            print(f'Identified possible states:{state} for sensors {sensors}')