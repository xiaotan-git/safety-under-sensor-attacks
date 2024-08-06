from typing import List
import itertools
import numpy as np
from scipy import linalg
# import mpmath as mpm
import control as ct

# sampling rate
TS = 0.05
EPS = 1e-6

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
    a = np.array([a1, a2, a3]) or a = np.array([[a1, a2, a3]]) or a = np.array([[a1], [a2], [a3]])
    a = right_shift_row_array(a,2) 
    # result 
    a = np.array([[a2,a3,a1]])
    '''
    a.reshape(1,-1)
    to_shift = a[0,-shift_amount:]
    a_shift = np.concatenate((to_shift,a[0,0:-shift_amount]))
    a_shift.reshape(1,-1)
    return a_shift

class SSProblem():
    '''
    This class defines system model, input and measurement data, etc for SSR problem
    
    Conventions: 
    A, B, C, D: state-space system matrices,  2d-array
    n,m,p,s,io_length: integers
    input_sequence, output_sequence: 2d-array. Each row denotes input at one time instant. 
    according to (5) in the paper, input output have the same length.
    output_sequence[-1,:] is the most recent input/output at current time  t
    input_sequence[-1,:] is the input to be determined, and must be all zero when initializes.
    input_sequence[0,:]/output_sequence[0,:] is the earliest input/output at time  t-io_length+1 or 0
    '''
    def __init__(self, dtsys_a, dtsys_b, dtsys_c, dtsys_d, output_sequence, attack_sensor_count=2, 
                 input_sequence = None,measurement_noise_level= None, is_sub_ssr = False) -> None:

        self.A = dtsys_a
        self.B = dtsys_b
        self.C = dtsys_c
        self.D = dtsys_d
        self.io_length = output_sequence.shape[0]
        self.noise_level = measurement_noise_level
        self.is_sub_ssr = is_sub_ssr

        self.n = np.shape(dtsys_a)[0] # dim of states
        self.m = np.shape(dtsys_b)[1] # dim of inputs
        self.p = np.shape(dtsys_c)[0] # no. of sensors
        self.s = attack_sensor_count # max no. of attacked sensors

        if input_sequence is not None:
            self.u_seq = input_sequence
            self.tilde_y_his = output_sequence
        else:
            self.u_seq = np.zeros((self.io_length,self.m))
            self.y_his = output_sequence   


        assert self.n == np.shape(self.A)[0]
        assert self.n == np.shape(self.A)[1]
        assert self.n == np.shape(self.B)[0]
        assert self.m == np.shape(self.B)[1]
        assert self.p == np.shape(self.C)[0]
        assert self.n == np.shape(self.C)[1]

        assert self.io_length == np.shape(self.u_seq)[0]
        assert self.m == np.shape(self.u_seq)[1]
        assert self.io_length == np.shape(output_sequence)[0]
        assert self.p == np.shape(output_sequence)[1]

    @staticmethod
    def convert_ct_to_dt(Ac, Bc, Cc, Dc, ts):
        '''
        From continuous time system (Ac, Bc, Cc, Dc) to discrete-time system (A,B,C,D) with ZOH discretization scheme 
        '''
        sys = ct.ss(Ac, Bc, Cc, Dc)
        discrete_sys = sys.sample(ts,method = 'zoh')
        return discrete_sys.A, discrete_sys.B, discrete_sys.C, discrete_sys.D, 

    @staticmethod
    def update_state_one_step(dtsys_a, dtsys_b, state_array,ut):
        '''
        This method propagates an array of system states xt forward in time given an input ut

        '''
        # xt is of dimension n \times no. of possible states
        n = dtsys_a.shape[0]
        m = dtsys_b.shape[1]
        ut.reshape(m,1)
        if state_array.ndim!=2:
            raise KeyError('When propagating state, state_array should be of 2d array')
        if state_array.shape[0] != n:
            raise KeyError('When propagating state, state_array should be of dimension n times number of possible states')
        x_new = dtsys_a@state_array + (dtsys_b@ut).reshape(n,1) #  ().reshape(n,1) for broadcasting
        return x_new
    
    @classmethod    
    def update_state(cls,dtsys_a,dtsys_b,state_array,u_seq):
        '''
        This method updates an array of system states xt given an input sequence u_seq. 
        x:  (n,k), must be a 2d-array
        u_seq: (t,m) array for multiple steps, (m,), (m,1), (1,m) array for one step update
        '''
        m = dtsys_b.shape[1]
        duration, remainder = divmod(u_seq.size, m)
        if remainder != 0:
            raise ValueError("Number of inputs divided by input size must be an integer")
        else:
            x_old = state_array
            if duration ==1:
                x_new = cls.update_state_one_step(dtsys_a, dtsys_b,x_old,u_seq)
            else:
                for t in range(duration):
                    x_new = cls.update_state_one_step(dtsys_a, dtsys_b,x_old,u_seq[t,:])
                    x_old = x_new
                    # print(f'x_new: {x_new}')
        return x_new
    
    @classmethod         
    def generate_attack_measurement(cls,dtsys_a, dtsys_b, dtsys_c, dtsys_d,sensor_indexed_init_state_list,
                                    s = 2, is_noisy = True, noise_level = 0.001, io_length = None,has_u_seq = True, u_seq = None):
        '''
        Generate the worst-case attack by assuming that the attacker tries to confuse the system with some states. 
        The correspondence between sensors and states is given in att_dic

        example:
        x1 = np.array([[10],[30]])
        x2 = np.array([[20],[30]])
        # 4 sensors
        sensor_indexed_init_state_list= [x1, x2, x1, x2]
        '''
        n = dtsys_a.shape[0]
        m = dtsys_b.shape[1]
        p = dtsys_c.shape[0]

        # measurement steps
        if io_length is None:
            io_length = n
        # input sequence generation if not given
        if has_u_seq and u_seq is None:
        # random input_sequence, oldest input comes first, e.g., u_seq[0,:] -> u(0) 
            u_seq = np.random.uniform(-5,5,(io_length,m))
            u_seq[-1,:] = 0.0
        if not has_u_seq:
            u_seq = np.zeros((io_length,m))

        init_states = np.hstack(sensor_indexed_init_state_list)
        xt_new = init_states
        tilde_y_his = np.zeros((io_length,p))
        for t in range(io_length):
            for i in range(p):
                corres_state = xt_new[:,i:i+1]
                # print(f'sensor {i}, corres. state {corres_state}')
                Ci = dtsys_c[i:i+1,:] # 2D array
                Di = dtsys_d[i:i+1,:] # 2D array
                yi_t = Ci@corres_state + Di@(u_seq[t,:].reshape(m,1))
                if not yi_t.size == 1:
                    print(f'Ci: {Ci}')
                    print(f'Di: {Di}')
                    print(f'corres_state: {corres_state}')
                    print(f'yi_t:{yi_t}')
                tilde_y_his[t,i] = yi_t # entry-wise assignment
            # update state to next time instant
            xt_old = xt_new
            xt_new = cls.update_state(dtsys_a,dtsys_b,xt_old,u_seq[t,:])

        if is_noisy:
            tilde_y_his = tilde_y_his + np.random.normal(0,float(noise_level),(io_length,p))
        else:
            noise_level = 0.0


        return u_seq, tilde_y_his, noise_level

class SecureStateReconstruct():
    '''
    This class implements different SSR algorithms and yields possible states and corresponding sensors
    '''
    def __init__(self, ss_problem:SSProblem,possible_comb = None) -> None:
        self.problem = ss_problem
        # 3d narray, shape (io_length, n, p)
        self.obser = self.construct_observability_matrices()

        # possible healthy sensor combinations
        if possible_comb is None:
            num_healthy_sensors = ss_problem.p-ss_problem.s
            self.possible_comb = nchoosek([i for i in range(ss_problem.p)],num_healthy_sensors)
        else:
            self.possible_comb = possible_comb

        # check if the clean measurement exists
        if hasattr(ss_problem,'y_his'):
            self.y_his = ss_problem.y_his
        else:
            # 2d narray, shape (io_length, p). Definition per (5)
            self.y_his = self.construct_clean_measurement()

    def construct_observability_matrices(self):
        ss_problem = self.problem
        obser_matrix_array = np.zeros((ss_problem.io_length,ss_problem.n,ss_problem.p))

        A = ss_problem.A
        for i in range(ss_problem.p):
            Ci = ss_problem.C[i:i+1,:]
            obser_i = Ci@linalg.fractional_matrix_power(A,0)
            for t in range(1,ss_problem.io_length):
                new_row = Ci@linalg.fractional_matrix_power(A,t)
                obser_i = np.vstack((obser_i,new_row))
            obser_matrix_array[:,:,i:i+1] = obser_i.reshape(ss_problem.io_length,ss_problem.n,1)
        return obser_matrix_array

    def construct_clean_measurement(self):
        ss_problem = self.problem
        tilde_y = ss_problem.tilde_y_his
        u_seq = ss_problem.u_seq
        io_length = ss_problem.io_length
        # Check (5) for definition
        # u_list = [input at time t_now - io_length+1, input at time t_now -io_length+2, ..., input at time t_now]
        u_list = [u_seq[t,:] for t in range(io_length)]
        u_vec = np.vstack(u_list).reshape(-1,1) # 2d column array

        # yi
        yi_list = []
        for i in range(ss_problem.p):
            tilde_yi = tilde_y[:,i:i+1] #2d array of dimension (io_length,1)
            fi = self.construct_fi(i) #2d array of dimension (io_length,io_length*m)
            yi = tilde_yi - fi@u_vec #2d array of dimension (io_length,1)
            yi_list.append(yi)
        
        # y_his: shape (io_length, p) = [y1 y2 ... yp] for p sensors, and yi is a column 2d array [yi(0); yi(1); .... yi(io_length-1)] 
        y_his = np.hstack(yi_list)
        return y_his
    
    def construct_fi(self,i):
        '''
        construct Fi according to Eq. 5 in the paper
        '''
        Ci = self.problem.C[i,:]
        A = self.problem.A
        B = self.problem.B
        io_length = self.problem.io_length
        m = self.problem.m
       
        fi = np.zeros((io_length,io_length*m))
        # note that row count of Fi  =  io_length. The first row is zeros. 
        # see (5) for definition
        for t in range(1,io_length):
            fi[t:t+1,:] = right_shift_row_array(fi[t-1:t,:],m)
            # here for t-th row, t = 1,...,io_length -1, add the left most element Ci A^t B. 
            fi[t:t+1,0:m] = Ci @ linalg.fractional_matrix_power(A,t-1) @ B 
        return fi
       
    # def construct_y_his_vec(self,comb):
    #     measure_vec_list = []
    #     for i in comb:
    #         measure_i = self.y_his[:,i:i+1]
    #         measure_vec_list.append(measure_i)
    #     measure_vec = np.vstack(measure_vec_list)
    #     return measure_vec

    def solve_initial_state(self,error_bound = 1):
        '''
        The method solves a given SSR problem and yields possible initial states, currently in a brute-force approach. 
        '''
        if self.problem.is_sub_ssr:
            raise Exception('It is a sub-ssr problem. Use solve_initial_state_subssr method instead')
        
        possible_states_list = []
        corresp_sensors_list = []
        residuals_list = []
        for comb in self.possible_comb:
            # recall obser is in the shape of (io_length, n, p)
            obser_matrix = self.vstack_comb(self.obser,comb)
            # print(f'obser_matrix for comb {comb} is \n {obser_matrix}')
            # recall y_his is in the shape of (io_length, p)
            measure_vec = self.vstack_comb(self.y_his,comb)
            # print(f'corresponding measurement is \n {measure_vec}')

            # print(f'Observation matrix shape {obser_matrix.shape}, measure_vec shape {measure_vec.shape}')
            state, residuals, rank, _ = linalg.lstsq(obser_matrix,measure_vec)

            if len(residuals)<1:
                # print(f'combinations: {comb}')
                residuals = linalg.norm(obser_matrix@state - measure_vec,ord=2)**2
            else:
                residuals = residuals.item()

            if residuals <error_bound:
                possible_states_list.append(state)
                corresp_sensors_list.append(comb)
                residuals_list.append(residuals)
                # print(f'residuals: {residuals}')
                # if rank < obser_matrix.shape[1]:
                    # print(f'Warning: observation matrix for sensors in {comb} is of deficient rank {rank}.')

        if len(possible_states_list)>0:
            # here we remove the states that yields 100x smallest residual
            residual_min = min(residuals_list)
            # print(f'residual min is {residual_min}')
            # comb_list = [i for i in range(len(residuals_list)) if residuals_list[i]<10*residual_min]
            possible_states_list = [possible_states_list[index] for index in range(len(residuals_list)) ]
            corresp_sensors_list = [corresp_sensors_list[index] for index in range(len(residuals_list)) ]
            possible_states = np.hstack(possible_states_list)
            corresp_sensors = np.array(corresp_sensors_list)
        else:
            possible_states = None
            corresp_sensors = None
            print('No possible state found. Consider relax the error bound')

        return possible_states, corresp_sensors, residuals_list
    
    def solve(self,error_bound = 1):
        # Solves for current states
        possible_states, corresp_sensors, residuals_list = self.solve_initial_state(error_bound)
        current_states_list = []
        if possible_states is not None: 
            for ind in range(possible_states.shape[1]):
                init_state = possible_states[:,ind:ind+1] # must be 2d and state.shape[0] must be n
                curr_state = self.problem.update_state(self.problem.A, self.problem.B, init_state,self.problem.u_seq)
                current_states_list.append(curr_state)
            current_states = np.hstack(current_states_list)
        else:
            current_states = None
        return current_states, corresp_sensors, residuals_list

    def generate_subssr_data(self):
        '''
        generate data A^(j), C_i, Y_i^j for sensor i = 1,...,p and generalized eigenspace V^j, j = 1,...,r
        subprob_a, subprob_c, subprob_y = self.gen_subssr_data(y_his)
        A^(j), C_i, Y_i^j = subprob_a[:,:,j], subprob_c[i,:], subprob_y[:,i,j]

        For more details: Y. Mao, A. Mitra, S. Sundaram, and P. Tabuada, “On the computational complexity of the secure state-reconstruction problem,” Automatica, vol. 136, p. 110083, 2022
        '''
        eigval_a = linalg.eigvals(self.problem.A)
        unique_eigvals, counts = np.unique(eigval_a,return_counts = True)
        # total number of subspaces V^j, j = 1,..., r
        r = len(unique_eigvals)

        generalized_eigenspace_list = self._compute_generalized_eigenspace_list(unique_eigvals, counts) # for a list of V^j

        # Observation projection tilde{P}_ij per sensor and per subspace
        # sensor_indexed_observation_subspaces[i] is a list of observation subspace O_i (V^j), j = 1,...,r
        sensor_indexed_observation_subspaces = self._compute_sensor_indexed_observation_subspaces(
            generalized_eigenspace_list
        )

        subprob_a_list = []
        subprob_y_new_list = []
        project_obs_new_list = []
        # list of lists, proj_obs_ij_list[j][i] gives tilde_P_ij
        proj_obs_ij_list = []
        for j in range(r):
            # for the generalized eigenspace V^j
            proj_gs_j = self.construct_proj(generalized_eigenspace_list,j) # This is P_j: R^n -> V^j

            # construct subsystem data A_j, C_j, Y_j for subspace V^j
            proj_a_j = proj_gs_j@self.problem.A # (n,n) 2d array
            subprob_a_list.append(proj_a_j)
            
            # Observ projection per sensor for subspace V^j
            # proj_y_his_j_new: (io_length,p) 2-Narray sensor-indexed projected y_his for jth subspace
            # project_obs_j: (io_length*p,io_length*p) 2-Narray sensor-indexed projection matrix from O(R^n) -> O(V^j) 
            proj_y_his_j_new, project_obs_j, proj_obs_j_list = self._compute_proj_y_his_j(sensor_indexed_observation_subspaces,j)
            subprob_y_new_list.append(proj_y_his_j_new)
            project_obs_new_list.append(project_obs_j)
            proj_obs_ij_list.append(proj_obs_j_list)
        
        # 3d narray with dimension (n,n,r)
        subprob_a = np.dstack(subprob_a_list)
        # 3d naaray with dimension (io_length,p,r)
        subprob_y = np.dstack(subprob_y_new_list)

        ## *************Not used, only for testing if the projected subproblem is correct************
        # Here we compute for a list of O(V^j) and the projection tilde_P_j: O(R^n) -> O(V^j). 
        # obser_full = self.vstack_comb(self.obser) # (O_1, O_2, ..., O_p) stacked vertically
        # observable_space_list = [] 
        # subprob_y_list= []
        # for j in range(r):
        #     obser_subspace = obser_full@generalized_eigenspace_list[j]
        #     observable_space_list.append(obser_subspace)
        # project_obs_list = []
        # for j in range(r):
        #     proj_obs_j = self.construct_proj(observable_space_list,j) # This is 
        #     project_obs_list.append(proj_obs_j)
        #     y_his_vec = self.vstack_comb(self.y_his) # recall y_his is (io_length, p)
        #     proj_y_his_j_vec =  proj_obs_j@y_his_vec # 2d column array, [y1(0), y1(1), ...,y1(io_length-1), y2(0), ....]
        #     proj_y_his_j = self.unvstack(proj_y_his_j_vec,self.problem.io_length)
        #     proj_y_his_j = np.reshape(proj_y_his_j_vec,(self.problem.io_length,self.problem.p),order='F')
        #     subprob_y_list.append(proj_y_his_j)
        # subprob_y = np.dstack(subprob_y_list)

        # print('--------------with tilde_Pj-----------')
        # self._test_projected_subproblem_data(r,subprob_a,project_obs_list)
        # print('--------------with tilde_P_ij---------')
        # np.random.seed(100)
        # x = np.random.normal(3,3,size=(self.problem.n,1))
        # self._test_projected_subproblem_data(x,r,subprob_a,project_obs_new_list,project_obs_list,proj_obs_ij_list)
        # self._test2(x,r,project_obs_new_list,proj_obs_ij_list)
        # ************************************************************************************

        # remove imaginary part
        # imaginary part appears only if self.problem.A has complex eigenvalues
        if linalg.norm(subprob_a - subprob_a.real)<= EPS:
            subprob_a = subprob_a.real
        if linalg.norm(subprob_y - subprob_y.real)<= EPS:
            subprob_y = subprob_y.real
        return subprob_a, self.problem.C, subprob_y
    
    def _compute_generalized_eigenspace_list(self,unique_eigvals, counts):
        generalized_eigenspace_list = [] # for a list of V^j
        r = len(unique_eigvals)
        for j in range(r):
            eigval =  unique_eigvals[j]
            am = counts[j]
            generalized_eigspace = self.compute_generalized_eigenspace(eigval,am, self.problem.A)
            generalized_eigenspace_list.append(generalized_eigspace)
        return generalized_eigenspace_list
    
    def _compute_sensor_indexed_observation_subspaces(self,generalized_eigenspace_list):
        r = len(generalized_eigenspace_list)
        sensor_indexed_observation_subspaces = []
        for i in range(self.problem.p):
            observ_space_sensor_i_list = []
            obser_i = self.obser[:,:,i]
            for j in range(r):
                generalized_eigspace = generalized_eigenspace_list[j]
                observ_space_ij =  obser_i@generalized_eigspace #io-length*dim(Vj) 2d array
                observ_space_sensor_i_list.append(observ_space_ij)
            sensor_indexed_observation_subspaces.append(observ_space_sensor_i_list)
        return sensor_indexed_observation_subspaces

    def _compute_proj_y_his_j(self,sensor_indexed_observation_subspaces,j):
        y_ij_list = []
        # this list iterates over all i given a subspace j
        proj_obs_j_list = []
        for i in range(self.problem.p):
            observ_space_sensor_i_list = sensor_indexed_observation_subspaces[i]
            # print(f'observ_space_ij_list:{observ_space_sensor_i_list}')
            proj_obs_ji = self.construct_proj(observ_space_sensor_i_list,j)  # This is tilde_P_ij: Oi(R^n) -> Oi(V^j)
            Yi = self.y_his[:,i:i+1]
            y_ij = proj_obs_ji@Yi
            y_ij_list.append(y_ij) # append over all i
            proj_obs_j_list.append(proj_obs_ji)
        
        proj_y_his_j = np.hstack(y_ij_list)
        project_obs_j = linalg.block_diag(*proj_obs_j_list)
        # print(f'project_obs_j shape :{project_obs_j.shape}')
        return proj_y_his_j,project_obs_j, proj_obs_j_list
    
    def _test_projected_subproblem_data(self,x,r,subprob_a,project_obs_list1,project_obs_list2,proj_obs_ij_list):
        # project_obs_list1: list of bar_P_j, the blockdiagnal of tilde_pij over i
        # project_obs_list2: list of tilde_P_j, the projection from O(R^n) -> O(V^j)
        # proj_obs_ij_list: list of lists, proj_obs_ij_list[j][i] the projection from Oi(R^n) -> Oi(V^j)
        obser_full = self.vstack_comb(self.obser)
        # np.random.seed(100)
        # x = np.random.normal(3,3,size=(self.problem.n,1))
        ax = self.problem.A@x
        y = obser_full@x

        ax_prime = np.zeros((self.problem.n,1))
        y_prime1 = np.zeros((self.problem.io_length*self.problem.p,1))
        y_prime2 = np.zeros((self.problem.io_length*self.problem.p,1))
        y_prime3 = np.zeros((self.problem.io_length*self.problem.p,1))

        for j in range(r):
            ax_j = subprob_a[:,:,j]@x
            ax_prime = ax_prime + ax_j
            # calculation 1: bar_P_j
            projection_obs_j1 = project_obs_list1[j]
            y_j1 = projection_obs_j1@y
            y_prime1 = y_prime1 + y_j1
            # calculation 2: tilde_P_j
            projection_obs_j2 = project_obs_list2[j]
            y_j2 = projection_obs_j2@y
            y_prime2 = y_prime2 + y_j2
            # calculation 3: tilde_P_ij
            pij_oi_list = []
            for i in range(self.problem.p):
                pij = proj_obs_ij_list[j][i]
                pij.reshape(1,-1)
                oi = self.obser[:,:,i]
                pij_oi = pij@oi
                pij_oi_list.append(pij_oi)
            rhs_mat = np.vstack(pij_oi_list)
            y_j3 = rhs_mat@x
            y_prime3 = y_prime3 + y_j3

            print(f'--------------j = {j} ---------')
            # print(f'bar_P_j Ox: {y_j1}')
            # print(f'tilde_P_j Ox: {y_j2}')
            print(f'difference norm (bar_P_j case   -  tilde_P_j case): {np.linalg.norm(y_j1- y_j2)}')
            print(f'difference norm (bar_P_j case  -  tilde_P_ij case): {np.linalg.norm(y_j1- y_j3)}')
            print(f'difference norm (tilde_P_j case - tilde_P_ij case): {np.linalg.norm(y_j2- y_j3)}')

        assert linalg.norm(ax - ax_prime)<= 1e-6, 'subprob_a in subssr data generation fails simple example'
        # if linalg.norm(y - y_prime1)> 1e-3: 
            # print(f'Warning: project_obs_list in subssr data generation example has an error {linalg.norm(y - y_prime1)} \n\n')
        print(f'total y has an error              (y - bar_P_j case): {linalg.norm(y - y_prime1)} ')
        print(f'total y has an error            (y - tilde_P_j case):  {linalg.norm(y - y_prime2)} ')
        print(f'total y has an error           (y - tilde_P_ij case):  {linalg.norm(y - y_prime3)} ')
        print(f'total y has an error (bar_P_j case - tilde_P_j case): {linalg.norm(y_prime1 - y_prime2)} ')
        # print(f'y: {y}')

    def _test2(self,x,r,project_obs_list,proj_obs_ij_list):
        obser_full = self.vstack_comb(self.obser)
        # np.random.seed(100)
        # x = np.random.normal(3,3,size=(self.problem.n,1))
        for j in range(r):
            P_bar_j = project_obs_list[j]
            lhs = P_bar_j@obser_full@x

            pij_oi_list = []
            for i in range(self.problem.p):
                pij = proj_obs_ij_list[j][i]
                pij.reshape(1,-1)
                oi = self.obser[:,:,i]
                pij_oi = pij@oi
                pij_oi_list.append(pij_oi)
            rhs_mat = np.vstack(pij_oi_list)
            rhs = rhs_mat@x

            if np.linalg.norm(lhs-rhs)>1e-4:
                print('-------------------------------------')
                print(f'subspace j:{j}, error: {np.linalg.norm(lhs-rhs)}')
                # print(f'lhs: {lhs}, \n rhs: {rhs}')
                print(f'mat error: {P_bar_j@obser_full-rhs_mat}')




    def solve_initial_state_subssr(self,eoi,subspace,error_bound = 1):
        '''
        It solves initial states for sub-SSR problem. Currently using voting to determine the initial state.

        Consider both brute-force approach and voting approach for subssr problem

        For now we only consider gm = 1 case and individual sensors w.r.t. which the system is measurable in that subspace

        eoi -- eigenvalue observability index. Same as soi sparse obser. index if gm = 1 for all eigenvalues.
        '''
        possible_states_list = []
        corresp_sensors_list = []
        attacked_sensors_list = []
        residuals_list = []

        states_to_vote_list = []
        sensors_list = []
        for i in range(self.problem.p):
            # recall obser is in the shape of (io_length, n, p)
            obser_matrix = self.obser[:,:,i]
            # print(f'obser_matrix for comb {comb} is \n {obser_matrix}')
            # recall y_his is in the shape of (io_length, p)
            measure_vec =self.y_his[:,i:i+1]

            # solving the equality Oij xj = Yij. 
            # j does not show explicitly because this is a subssr problem instance
            state, residuals, rank= self.solve_lstsq_from_subspace(obser_matrix,measure_vec,subspace)

            if residuals <error_bound:
                states_to_vote_list.append(state)
                sensors_list.append(i)
                residuals_list.append(residuals)

        # voting on states_to_vote_list
        # sorted_states = [[0,(0,2,3)],[1,(1,4)] ,... ] [state, corresponding sensors]
        sorted_states =  self.vote_on_states(states_to_vote_list, error_bound=1e-2)
        # print(f'sorted_states, {sorted_states}')
        # populating possible_states and corresp_sensors
        for state_ind in sorted_states:
            # voting criterion: votes \ge p' - s, where p' is the eigenvalue observability + 1
            if len(sorted_states[state_ind])>= eoi+1 - self.problem.s:
                states_list = [states_to_vote_list[i] for i in sorted_states[state_ind]]
                # averaging out the "same" states
                average_state = np.average(np.hstack(states_list),axis=1).reshape(-1,1)
                possible_states_list.append(average_state)
                corresp_sensors_list.append(sorted_states[state_ind])
        # print(f'possible_states_list: {possible_states_list}')

        if len(possible_states_list) == 0:
            print('No possible state found for sub-ssr problem. Consider relax the voting range bound')
            print(f'sorted_states: {sorted_states}')
            print(f'states_to_vote_list:{states_to_vote_list}')
            return None, None
        possible_states = np.hstack(possible_states_list)

        for k in range(len(possible_states_list)):
            state = possible_states_list[k]
            corresp_sensors = corresp_sensors_list[k]
            # by introducing attacked_sensors_candi list, we avoid check for all sensors
            attacked_sensors_candi = [i for i in range(self.problem.p) if i not in corresp_sensors]
            attacked_sensors = []
            for i in attacked_sensors_candi:
                Oi = self.obser[:,:,i]
                yij = self.y_his[:,i:i+1]
                # this gives me a hard time
                if linalg.norm(yij - Oi@(state.reshape(-1,1)))>1e-1:
                    attacked_sensors.append(i)
            attacked_sensors_list.append(attacked_sensors)

        return possible_states, attacked_sensors_list
    

    def vote_on_states(self,states_to_vote_list,error_bound=1e-3):
        '''
        returns sorted_states, voting results on states_to_vote_list and its corresponding sensors_list

        '''
        def is_same_state(state1,state2):
            return linalg.norm(state1 - state2)<error_bound
        counts = {}
        # e.g., counts = {0:3, 1:2} meaning the first state has 3 votes, the second state has 2 votes
        counts[0] = [ ]

        for ind in range(len(states_to_vote_list)):
            state = states_to_vote_list[ind]
            is_checked = False
            for state_ind in list(counts):
                st = states_to_vote_list[state_ind]
                if is_same_state(state,st) and not is_checked:
                    # counts[state_ind] =  counts[state_ind] + 1
                    counts[state_ind].append(ind)
                    is_checked = True
            if not is_checked:
                counts[ind] = [ind]

        # Sort elements by count in descending order
        sorted_states = dict(sorted(counts.items(), key=lambda x: len(x), reverse=True))
        return sorted_states
    
    def compose_subspace_states(self,subspace_states_sensors_list):
        possible_states = []
        corresp_sensors = []
        combinations = list(itertools.product(*subspace_states_sensors_list))
        for comb in combinations:
            try:
                total_state = np.zeros(shape = comb[0][0].shape)
            except:
                print(f'comb[0]: {comb[0]}')
            total_attacked_sensors = []
            for item in comb:
                total_state = total_state + item[0]
                total_attacked_sensors.extend(item[1])
            total_unique_attacked_sensors = list(set(total_attacked_sensors))
            if len(total_unique_attacked_sensors)<=self.problem.s:
                possible_states.append(total_state)
                corresp_sensor = [sensor for sensor in range(self.problem.p) if sensor not in total_unique_attacked_sensors]
                corresp_sensors.append(corresp_sensor)
        
        if not possible_states:
            return None, None
        # print('-----------------------------------')
        # print(f'possible_states: {possible_states}')
        # print(f'corresp_sensors:{corresp_sensors}')
        return possible_states,corresp_sensors


    @classmethod
    def vstack_comb(cls,array_to_stack:np.ndarray,comb:list = None,axis:int = None):
        '''
        vertically stack numpy array that is sliced along **axis** with some **combination**, 
        default axis: last axis; default comb: full indices    

        a = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5,2])
        b = a.reshape(2,2,-1)
        c = SecureStateReconstruct.vstack_comb(b,comb = [1,2],axis = 2) 
        '''
        if axis is None:
            axis = array_to_stack.ndim-1 # last dimension
        if comb is None:
            p = array_to_stack.shape[axis]
            comb = [i for i in range(p)]
        temp_list = []
        for i in comb:
            temp_i = cls.slice_along_axis(array_to_stack,i,axis)
            if temp_i.ndim == 1:
                temp_i = temp_i.reshape(-1,1) # so temp_i becomes 2d column array
            temp_list.append(temp_i)
        stacked_array = np.vstack(temp_list)
        return stacked_array
        
    @staticmethod
    def unvstack(array_to_unstack:np.ndarray,row_count_for_a_slice:int):
        '''
        undo the vstack operation
        array_to_unstack: 2d or 3d array

        Example:
        a = np.array(range(12))
        a = a.reshape(-1,1)
        # print(f'a before unvstack: {a}')
        io_len = 4
        a_prime = SecureStateReconstruct.unvstack(a,io_len)
        # a_prime = [[ 0  4  8]
        # [ 1  5  9]
        # [ 2  6 10]
        # [ 3  7 11]]

        a = np.array(range(24))
        a = a.reshape(-1,2)
        print(f'a before unvstack: {a}')
        io_len = 4
        a_prime = SecureStateReconstruct.unvstack(a,io_len)
        print(f' {a_prime[:,:,0]}') # [[0, 1], [2, 3],[4,5],[6,7]]

        '''
        rows = array_to_unstack.shape[0]
        quotient, remainder = divmod(rows, row_count_for_a_slice) 
        if remainder !=0:
            print('Warning:unable to unvstack the array')
        temp_lst = []
        for i in range(quotient):
            temp_lst.append(array_to_unstack[i*row_count_for_a_slice:(i+1)*row_count_for_a_slice,:])
        if array_to_unstack.shape[1] == 1:
            array_unstacked = np.hstack(temp_lst)
        else:
            array_unstacked = np.dstack(temp_lst)

        return array_unstacked
    
    @staticmethod
    def slice_along_axis(array, j, k):
        """
        Extract the j-th slice along the k-th axis of a d-dimensional array.

        Parameters:
        array (np.ndarray): The input d-dimensional array.
        j (int): The index of the slice to extract.
        k (int): The axis along which to extract the slice.

        Returns:
        np.ndarray: The extracted slice.
        """
        # Create a full slice (:) for each dimension
        index = [slice(None)] * array.ndim
        
        # Replace the slice for the k-th axis with the specific index j
        index[k] = j
        
        # Convert the list to a tuple and use it to slice the array
        return array[tuple(index)]
    
    @classmethod
    def compute_sys_eigenproperty(cls,A,output_gen_eigenspace = True):
        '''
        compute eigenvalue, (generalized) eigenspace related properties given system transition matrix A

        return unique_eigvals, (generalized) eigenspace, am_list, gm_list
        '''
        eigval_a, eigenspace_a = linalg.eig(A,right=True)
        unique_eigvals, counts = np.unique(eigval_a,return_counts = True)
        # total number of subspaces V^j, j = 1,..., r
        r = len(unique_eigvals)

        generalized_eigenspace_list = [] # for a list of V^j
        eigenspace_list = [] # for a list of eigenspace
        am_list = [] # a list for algebraic mutiplicities
        gm_list = [] # a list for geometric multiplicities
        for j in range(r):
            eigval =  unique_eigvals[j]
            am = counts[j]
            generalized_eigspace = cls.compute_generalized_eigenspace(eigval,am, A)

            if am == 1:
                # since am is always larger than gm
                gm = 1
                ind = sum(counts[0:j])
                eigenspace = eigenspace_a[:,ind]
            else:
                I_n = np.eye(A.shape[0])
                # this causes numerical issues
                eigenspace = linalg.null_space(A - eigval*I_n,rcond=EPS)
                gm = eigenspace.shape[1]
            
            generalized_eigenspace_list.append(generalized_eigspace)
            eigenspace_list.append(eigenspace)
            am_list.append(am)
            gm_list.append(gm)
        assert sum(am_list) == A.shape[0]
        assert sum(gm_list) <= A.shape[0]
        if output_gen_eigenspace:
            return unique_eigvals, generalized_eigenspace_list, am_list, gm_list
        else:
            return unique_eigvals, eigenspace_list, am_list, gm_list

    @classmethod
    def compute_sparse_observability(cls,A,C, is_scalar_measurement = True):
        '''
        compute sparse observability index given system matrices A,C. Currently only working for scalar measurement
        '''
        p = C.shape[0]
        soi = p
        for k in range(1,p+1):
            if cls.is_s_sparse_observability(A,C,k):
                pass
            else:
                return k-1 # removing k sensors fail the observability check

    @staticmethod
    def is_s_sparse_observability(A,C, s, is_scalar_measurement = True):
        '''
        check if given system matrices A,C is s-sparse observable. Currently only working for scalar measurement
        '''
        p = C.shape[0]
        if s == p:
            return False
        comb_list = nchoosek([i for i in range(p)], p - s)
        for comb in comb_list:
            C_comb = C[comb,:]
            # print(f'C_comb: {C_comb}')
            obser = ct.obsv(A,C_comb)
            # print(f's: {s}, np.linalg.matrix_rank(obser): {np.linalg.matrix_rank(obser)}')
            # this causes a numerical issue
            if np.linalg.matrix_rank(obser,tol=EPS) != A.shape[0]:
                return False

        return True

    @classmethod
    def compute_eigenvalue_observability(cls,A,C, is_scalar_measurement=True):
        '''
        compute eigenvalue observability index given system matrices A, C. Currently only working for scalar measurement
        '''
        unique_eigvals, eigenspace_list, am_list, gm_list = cls.compute_sys_eigenproperty(A)
        for i in range(len(gm_list)):
            if gm_list[i] != 1:
                print(f'Warning: eigenvalue observability is not well-defined for eigvalue {unique_eigvals[i]} due to geometric multiplicity {gm_list[i]}. Proceed to calculation anyway.')
        eigenvalue_observability_ind = C.shape[0] # start from p, keep decreasing
        for j in range(unique_eigvals.shape[0]):
            eigenval = unique_eigvals[j]
            ind_j = cls.compute_eigenvalue_obser_index_sensors(A,C,eigenval) # eigenvalue obser index for eigenvalue j
            eigenvalue_observability_ind = min(eigenvalue_observability_ind,ind_j)

        return eigenvalue_observability_ind
        
    @classmethod
    def compute_eigenvalue_obser_index_sensors(cls,A,C,eigenval_j,unique_eigvals = None,is_scalar_measurement=True):
        '''
        compute eigenvalue observability index for eigenval j given system matrices A, C. 
        i.e., it finds out how many sensors eigenval j is eigenvalue observable w.r.t. and then minus one to that number 
        Currently only working for scalar measurement
        '''
        eigenvalue_observ_ind = 0
        for i in range(C.shape[0]):
            C_i = C[i,:]
            if cls.is_eigenvalue_observable_onesensor(A,C_i,eigenval_j,unique_eigvals=unique_eigvals):
                eigenvalue_observ_ind = eigenvalue_observ_ind + 1
        return (eigenvalue_observ_ind-1) # due to the definition in the paper

    @classmethod
    def is_eigenvalue_observable_onesensor(cls,A,C_i,eigenval_j,unique_eigvals = None):
        '''
        check if eigenvalue_j is observable with respect to sensor i
        Note this is meaningful if and only if gm(A) = 1 for all eigenvalues
        '''
        eigenval_j = cls.find_closest_eigenvalue(A,eigenval_j)
        n = A.shape[0]
        I_n = np.eye(n)
        mat = np.vstack([A -eigenval_j*I_n, C_i] )
        # print(f'A: {A}, eigenval_j: {eigenval_j} ')
        # print(f'PBH test matrix: \n {mat}')
        rank = np.linalg.matrix_rank(mat)
        # print(f'rank is {rank}')

        if rank == n:
            return True
        else:
            return False

    @staticmethod
    def find_closest_eigenvalue(A,eigenval_to_test):
        '''
        return one eigenvalue of A that is closest to eigenval_to_test. 
        This step is mainly to reduce numerical trancation error
        '''
        unique_eigvals = linalg.eigvals(A)

        # check if eigenval is in unique_eigvals
        temp = abs(unique_eigvals - eigenval_to_test)
        if np.min(temp)<EPS:
            ind = np.argmin(temp)
            return unique_eigvals[ind] 
        else:
            print('Warning:eigenval_to_test is far from any eigenvalues of A')
            return None

    @staticmethod
    def compute_generalized_eigenspace(eigval, am, A):
        '''
        Calculates generalized eigenspace of A corresp. to eigval
        am -- algebraic multiplicity
        '''
        if am == 1:
            eigenvalues, eigenvectors = np.linalg.eig(A)
            # Find the index of the eigenvalue closest to the given one
            idx = np.argmin(np.abs(eigenvalues - eigval))
            # Return the corresponding eigenvector
            generalized_eigenspace = eigenvectors[:, idx:idx+1]
        else:
            eigval = SecureStateReconstruct.find_closest_eigenvalue(A,eigval)
            # linalg.null_space is numerically sensitive. rcond - relative conditional number
            temp_matrix = linalg.fractional_matrix_power(A - eigval*np.eye(A.shape[0]), am)
            generalized_eigenspace = linalg.null_space(temp_matrix,rcond = EPS)
            if generalized_eigenspace.shape[1] > am:
                generalized_eigenspace = linalg.null_space(temp_matrix,rcond = 1e-3)
        
        assert generalized_eigenspace.shape[1] == am, f'generalized_eigenspace.shape:{generalized_eigenspace.shape}, am: {am}'
        return generalized_eigenspace
        
    @staticmethod
    def construct_proj(subspace_list,j):
        '''
        construct the projection matrix from the whole space to the jth subspace, i.e., 
        v = sum_j vj, vj = Pj*v, vj \in V_j, the subspace j
        Moreover, we know Pj*spj = spj, Pj*spi = 0 . To calculate Pj, we choose a basis for the whole space 
        V = (sp1,...,spj,...,spr). Based on [0, ..., spj, ...., 0] = Pj* [sp1, ..., spj, ...., spr] and 
        V is full column rank, we know
        ************  Pj = [0, ..., spj, ...., 0] (V^T V)^{-1} V^T   **************

        j = 0,1,..., r-1
        '''
        full_space = np.hstack(subspace_list) #  = V = (sp1, sp2, ..., spr)
        full_space.astype(np.float128)
        rank_full_space = np.linalg.matrix_rank(full_space)
        
        assert rank_full_space == full_space.shape[1], f"rank_full_space:{rank_full_space}, \n full_space: \n {full_space}"
        
        # to construct [0, ..., spj, ...., 0] 
        tem_list = []
        for item in range(len(subspace_list)):
            subspace = subspace_list[j]
            if item != j:
                tem_list.append(np.zeros(subspace.shape))
            else:
                tem_list.append(subspace)
        
        lhd = np.hstack(tem_list)
        # It is suggested that linalg.inv is not reliable. See https://stackoverflow.com/questions/31256252/why-does-numpy-linalg-solve-offer-more-precise-matrix-inversions-than-numpy-li
        # proj_mat = lhd @ linalg.inv(full_space.T @ full_space) @ full_space.T 
        proj_mat = lhd @ linalg.pinv(full_space)
        proj_mat.astype(np.float64)
        return proj_mat
    
    @staticmethod
    def solve_lstsq_from_subspace(A,b,subspace,error_bound = 1e-4):
        '''
        Solve least square problem by searching solution from a subspace of R^n
        Ax = b, x \in V= span(v1,v2,...,vp) => A V y = b, y\in R^p and AV should be full column rank

        Default scipy.linalg.lstsq does not handle rank-deficient matrix well
        '''
        A_new = A@subspace
        y, residuals, rank, _= linalg.lstsq(A_new,b)
        y = y.reshape(-1,1)
        # residuals = linalg.norm(A_new@y - b)**2
        if residuals>error_bound:
            print(f'Warning: residual from solve_lstsq_from_subspace is {residuals}')
        return subspace@y, residuals, rank

if __name__ == "__main__":
    # define system model and measurement model
    Ac = np.array([[1, 0, 0, 0],[0, 2, 0, 0],[0,0,3,0],[0,0,0,4]])
    # Ac = np.random.normal(3,3,size=(4,4))
    Bc = np.array([[0, 0],[1, 0],[0, 0],[0,1]])
    Cc = np.array([[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]])
    Dc = np.zeros((Cc.shape[0],Bc.shape[1]))
    s = 0
    is_testing_subssr = False

    # generate discrete-time system
    dtsys_a, dtsys_b, dtsys_c, dtsys_d = SSProblem.convert_ct_to_dt(Ac,Bc,Cc,Dc,TS)
    # define input output data
    init_state1 = np.random.normal(2,2,size=(Ac.shape[0],1))
    init_state2 = 2*init_state1
    # u_seq = np.array([[1,1],[1,1],[1,1],[0,0]])
    # assume sensors 1,3,5 are under attack
    sensor_initial_states = [init_state2 if i < s else init_state1 for i in range(Cc.shape[0])]

    u_seq, tilde_y_his, noise_level = SSProblem.generate_attack_measurement(dtsys_a, dtsys_b, dtsys_c, dtsys_d,sensor_initial_states,
                                                                            s = s,is_noisy = False, noise_level=1e-8,u_seq = None)

    # construct a problem instance
    ss_problem = SSProblem(dtsys_a, dtsys_b, dtsys_c, dtsys_d, tilde_y_his, 
                           attack_sensor_count=s,input_sequence=u_seq,measurement_noise_level= noise_level )
    print(f'A: \n {ss_problem.A},  \n B:\n  {ss_problem.B}, \n C: \n {ss_problem.C}, \n D:\n {ss_problem.D}')

    # ssr_solution = SecureStateReconstruct(ss_problem,possible_comb=comb1)
    ssr_solution = SecureStateReconstruct(ss_problem)
    # possible_states,corresp_sensors, _ = ssr_solution.solve(error_bound = 1)
    possible_states,corresp_sensors, _ = ssr_solution.solve_initial_state(error_bound = 1e-3)

    if possible_states is not None:
        for ind in range(corresp_sensors.shape[0]):
            sensors = corresp_sensors[ind,:]
            state = possible_states[:,ind]
            print('--------------------------------------------------------------------')
            print(f'Identified possible states:{state} for sensors {sensors}')
            print(f'Estimated state - initial state 1 is \n {state.reshape(-1,1)-init_state1}')
            print(f'Estimated state - initial state 2 is \n {state.reshape(-1,1)-init_state2}')
    
    # decompostion method
    if is_testing_subssr:
        # sparse observability index
        soi = SecureStateReconstruct.compute_sparse_observability(ss_problem.A,ss_problem.C)
        # eigenvalue observability index
        eoi = SecureStateReconstruct.compute_eigenvalue_observability(ss_problem.A,ss_problem.C)
        print(f'The problem have a sparse observability index {soi}, eigenvalue observability index: {eoi}, attacked sensor count: {s}')
        # eigenspace related property
        unique_eigvals, generalized_eigenspace_list, am_list, gm_list = SecureStateReconstruct.compute_sys_eigenproperty(ss_problem.A)
        # print(f'Eigen properties. . \n generalized_eigenspace_list: \n {generalized_eigenspace_list} ')
        print(f'unique_eigvals: {unique_eigvals.T}, algebraic multiplicities: {am_list}, geometric multiplicities: {gm_list}')

        # decompose ssr to sub_ssr problems
        subprob_a, subprob_c, subprob_y = ssr_solution.generate_subssr_data()

        # give cbf parameters H, q, gamma, solve for input feasible region
        subspace_states_attacked_sensors_list = []
        for j in range(subprob_a.shape[2]):
            sub_a = subprob_a[:,:,j]
            sub_c = subprob_c
            sub_y_his = subprob_y[:,:,j]
            # the following approach to solve for sub-SSR problem is not correct. 
            # subproblem = SSProblem(sub_a, dtsys_b, sub_c, dtsys_d,sub_y_his, attack_sensor_count=s,measurement_noise_level=noise_level)
            # sub_solution = SecureStateReconstruct(subproblem)


            subproblem = SSProblem(sub_a, dtsys_b, sub_c, dtsys_d,sub_y_his, 
                                   attack_sensor_count=s,is_sub_ssr=True)
            sub_solution = SecureStateReconstruct(subproblem)
            # solution by voting
            states, attacked_sensors_list = sub_solution.solve_initial_state_subssr(eoi,generalized_eigenspace_list[j])

            if states is None:
                raise KeyError(f'state in subspace {j} is None. Check')
            states = states.transpose() # to zip
            subspace_states_attacked_sensors = list(zip(states,attacked_sensors_list))
            subspace_states_attacked_sensors_list.append(subspace_states_attacked_sensors) # [..., [xj,[0,1,2]], ...]
        
        full_state_list,corresp_sensors = sub_solution.compose_subspace_states(subspace_states_attacked_sensors_list)
        if full_state_list is not None:
            state_ind = 0
            for full_state in full_state_list:
                print('--------------------------------------------------------------------')
                print(f'Estimated state {state_ind} - initial state 1 is \n {full_state.reshape(-1,1)-init_state1}')
                print(f'Estimated state {state_ind} - initial state 2 is \n {full_state.reshape(-1,1)-init_state2}')
                state_ind = state_ind +1
    pass
