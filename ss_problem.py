from typing import List
import itertools
import numpy as np
from scipy import linalg
import control as ct

# sampling rate
TS = 0.4

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
    n,m,p,s,tspan: integers
    input_sequence, output_sequence: 2d-array. Each row denotes input at one time instant. 
    input_sequence[0,:]/output_sequence[0,:] is the earliest input/output at time  t-tspan+1
    input_sequence[-1,:]/output_sequence[-1,:] is the most recent input/output at time  t
    '''
    def __init__(self, dtsys_a, dtsys_b, dtsys_c, dtsys_d, output_sequence, 
                 attack_sensor_count=2, input_sequence = None,measurement_noise_level= None) -> None:

        self.A = dtsys_a
        self.B = dtsys_b
        self.C = dtsys_c
        self.D = dtsys_d
        self.tspan = output_sequence.shape[0]
        self.noise_level = measurement_noise_level

        self.n = np.shape(dtsys_a)[0] # dim of states
        self.m = np.shape(dtsys_b)[1] # dim of inputs
        self.p = np.shape(dtsys_c)[0] # no. of sensors
        self.s = attack_sensor_count # max no. of attacked sensors

        if input_sequence is not None:
            self.u_seq = input_sequence
            self.tilde_y_his = output_sequence
        else:
            self.u_seq = np.zeros((self.tspan,self.m))
            self.y_his = output_sequence   


        assert self.n == np.shape(self.A)[0]
        assert self.n == np.shape(self.A)[1]
        assert self.n == np.shape(self.B)[0]
        assert self.m == np.shape(self.B)[1]
        assert self.p == np.shape(self.C)[0]
        assert self.n == np.shape(self.C)[1]

        assert self.tspan == np.shape(self.u_seq)[0]
        assert self.m == np.shape(self.u_seq)[1]
        assert self.tspan == np.shape(output_sequence)[0]
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
        if state_array.shape[0] != n:
            state_array = state_array.transpose()
        ut.reshape(m,1)
        x_new = dtsys_a@state_array + (dtsys_b@ut).reshape(n,1) #  ().reshape(n,1) for broadcasting
        return x_new
    
    @classmethod    
    def update_state(cls,dtsys_a,dtsys_b,state_array,u_seq):
        '''
        This method updates an array of system states xt given an input sequence u_seq. 
        x: (n,), (n,1), (1,n) array
        u_seq: (t,m) array for multiple steps, (m,), (m,1), (1,m) array for one step update
        '''
        m = dtsys_b.shape[1]
        tspan, remainder = divmod(u_seq.size, m)
        if remainder != 0:
            raise ValueError("Number of inputs divided by input size must be an integer")
        else:
            x_old = state_array
            if tspan ==1:
                x_new = cls.update_state_one_step(dtsys_a, dtsys_b,x_old,u_seq)
            else:
                for t in range(tspan):
                    x_new = cls.update_state_one_step(dtsys_a, dtsys_b,x_old,u_seq[t,:])
                    x_old = x_new
        return x_new
    
    @classmethod         
    def generate_attack_measurement(cls,dtsys_a, dtsys_b, dtsys_c, dtsys_d,sensor_indexed_init_state_list,
                                    s = 2, is_noisy = True, noise_level = 0.001, tspan = None,u_seq = None):
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
        if tspan is None:
            tspan = n
        # input sequence
        if u_seq is None:
        # random input_sequence, oldest input comes first, e.g., u_seq[0,:] -> u(0) 
            u_seq = np.random.uniform(-5,5,(tspan,m))
            u_seq[-1,:] = 0.0

        init_states = np.hstack(sensor_indexed_init_state_list)
        xt_new = init_states
        tilde_y_his = np.zeros((tspan,p))
        for t in range(tspan):
            for i in range(p):
                corres_state = xt_new[:,i:i+1]
                # print(f'sensor {i}, corres. state {corres_state}')
                Ci = dtsys_c[i:i+1,:] # 2D array
                Di = dtsys_d[i:i+1,:] # 2D array
                yi_t = Ci@corres_state + Di@(u_seq[t,:].reshape(m,1))
                tilde_y_his[t,i] = yi_t # entry-wise assignment
            # update state to next time instant
            xt_old = xt_new
            xt_new = cls.update_state(dtsys_a,dtsys_b,xt_old,u_seq[t,:])

        if is_noisy:
            tilde_y_his = tilde_y_his + np.random.normal(0,float(noise_level),(tspan,p))


        return u_seq, tilde_y_his, noise_level

class SecureStateReconstruct():
    '''
    This class implements different SSR algorithms and yields possible states and corresponding sensors
    '''
    def __init__(self, ss_problem:SSProblem,possible_comb = None) -> None:
        self.problem = ss_problem
        # 3d narray, shape (tspan, n, p)
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
            # 2d narray, shape (tspan, p)
            self.y_his = self.construct_clean_measurement()

    def construct_observability_matrices(self):
        ss_problem = self.problem
        obser_matrix_array = np.zeros((ss_problem.tspan,ss_problem.n,ss_problem.p))

        A = ss_problem.A
        for i in range(ss_problem.p):
            Ci = ss_problem.C[i:i+1,:]
            obser_i = Ci@linalg.fractional_matrix_power(A,0)
            for t in range(1,ss_problem.tspan):
                new_row = Ci@linalg.fractional_matrix_power(A,t)
                obser_i = np.vstack((obser_i,new_row))
            obser_matrix_array[:,:,i:i+1] = obser_i.reshape(ss_problem.tspan,ss_problem.n,1)
            print(obser_matrix_array)
        return obser_matrix_array

    def construct_clean_measurement(self):
        ss_problem = self.problem
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
        
        # y_his: shape (tspan, p) = [y1 y2 ... yp] for p sensors, and yi is a column vector [yi(0); yi(1); .... yi(tspan-1)] 
        y_his = np.hstack(yi_list)
        return y_his
    
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
            fi[t:t+1,0:m] = Ci @ linalg.fractional_matrix_power(A,t) @ B 
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
        possible_states_list = []
        corresp_sensors_list = []
        residuals_list = []
        for comb in self.possible_comb:
            # recall obser is in the shape of (tspan, n, p)
            obser_matrix = self.vstack_comb(self.obser,comb)
            # recall y_his is in the shape of (tspan, p)
            measure_vec = self.vstack_comb(self.y_his,comb)

            # print(f'Observation matrix shape {obser_matrix.shape}, measure_vec shape {measure_vec.shape}')
            state, residuals, rank, _ = linalg.lstsq(obser_matrix,measure_vec)

            if len(residuals)<1:
                # print(f'combinations: {comb}')
                residuals = linalg.norm(obser_matrix@state - measure_vec,ord=2)**2

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
        # Solves for current states
        possible_states, corresp_sensors, corresp_sensors_list = self.solve_initial_state(error_bound)
        current_states_list = []
        for ind in range(possible_states.shape[1]):
            init_state = possible_states[:,ind]
            curr_state = self.problem.update_state(self.problem.A, self.problem.B, init_state,self.problem.u_seq)
            current_states_list.append(curr_state)
        current_states = np.hstack(current_states_list)
        return current_states, corresp_sensors, corresp_sensors_list

    def gen_subssr_data(self,y_his):
        '''
        generate data A^(j), C_i, Y_i^j for sensor i = 1,...,p and generalized eigenspace V^j, j = 1,...,r
        subprob_a, subprob_c, subprob_y = self.gen_subssr_data(y_his)
        A^(j), C_i, Y_i^j = subprob_a[:,:,j], subprob_c[i,:], subprob_y[:,i,j]
        '''
        eigval_a = linalg.eigvals(self.problem.A)
        unique_eigvals, counts = np.unique(eigval_a,return_counts = True)
        # total number of subspaces V^j, j = 1,..., r
        r = len(unique_eigvals)
        obser_full = self.vstack_comb(self.obser)

        geig_space_list = []
        gobser_space_list = []
        for j in range(r):
            eigval =  unique_eigvals[j]
            am = counts[j]
            geigspace = self.gen_eigenspace(eigval,am, self.problem.A)
            obser_subspace_vec_list = []
            for k in range(geigspace.shape[1]):
                obser_subspace_vec = obser_full@geigspace[:,k]
                obser_subspace_vec = obser_subspace_vec.reshape(-1,1) #N*1 2d array
                print(f'shape of obser_subspace_vec: {obser_subspace_vec.shape}')
                obser_subspace_vec_list.append(obser_subspace_vec)
            obser_subspace = np.hstack(obser_subspace_vec_list)
            print(f'shape of obser_subspace: {obser_subspace.shape}')

            geig_space_list.append(geigspace)
            gobser_space_list.append(obser_subspace)

        subprob_a_list = []
        subprob_y_list = []
        for j in range(r):
            # for the generalized eigenspace V^j
            proj_gs_j = self.construct_proj(geig_space_list,j) # This is P_j: R^n -> V^j
            proj_obs_j = self.construct_proj(gobser_space_list,j) # This is tilde_P_j: O(R^n) -> O(V^j)

            # construct a,c,y for subspace V^j
            proj_a_j = proj_gs_j@self.problem.A # 2d array
            y_his_vec = self.vstack_comb(self.y_his)
            print(f'shape of proj_obs_j: {proj_obs_j.shape}, shape of y_his_vec: {y_his_vec.shape}')
            proj_y_his_j_vec =  proj_obs_j@y_his_vec # 2d column array, [y1(0), y1(1), ...,y1(tspan-1), y2(0), ....]
            subprob_a_list.append(proj_a_j)
            proj_y_his_j = self.unvstack(proj_y_his_j_vec,self.problem.tspan)
            subprob_y_list.append(proj_y_his_j)

        subprob_a = np.dstack(subprob_a_list).real
        subprob_y = np.dstack(subprob_y_list).real

        assert linalg.norm(subprob_a - np.dstack(subprob_a_list))<= 1e-6
        assert linalg.norm(subprob_y - np.dstack(subprob_y_list))<= 1e-6
            
        return subprob_a, self.problem.C, subprob_y

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
        '''
        rows = array_to_unstack.shape[0]
        quotient, remainder = divmod(rows, row_count_for_a_slice) 
        if remainder !=0:
            Warning('unable to unvstack the array')
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

    @staticmethod
    def gen_eigenspace(eigval, am, A):
        '''
        Calculates generalized eigenspace of A corresp. to eigval
        am -- algebraic multiplicity
        '''
        temp_matrix = linalg.fractional_matrix_power(A - eigval*np.eye(A.shape[0]), am)
        generalized_eigenspace = linalg.null_space(temp_matrix)
        return generalized_eigenspace
        
    @staticmethod
    def construct_proj(subspace_list,j):
        '''
        construct the projection matrix from the whole space to the jth subspace, i.e., 
        v = sum_j vj, vj = Pj*v, vj \in subspace_j
        Moreover, we know Pj*spj = spj, Pj*spi = 0 . To calculate Pj, we choose a basis for the whole space 
        V = (sp1,...,spj,...,spr). Based on [0, ..., spj, ...., 0] = Pj* [sp1, ..., spj, ...., spr] and 
        V is full column rank, we know
        ************  Pj = [0, ..., spj, ...., 0] (V^T V)^{-1} V^T   **************
        '''
        full_space = np.hstack(subspace_list) #  (sp1, sp2, ..., spr)
        rank_full_space = np.linalg.matrix_rank(full_space)
        assert rank_full_space == full_space.shape[1]

        tem_list = []
        for item in range(len(subspace_list)):
            subspace = subspace_list[j]
            if item != j:
                tem_list.append(np.zeros(subspace.shape))
            else:
                tem_list.append(subspace)
        
        lhd = np.hstack(tem_list)

        proj_mat = lhd @ linalg.inv(full_space.T @ full_space) @ full_space.T 
        return proj_mat

if __name__ == "__main__":
    # define system model and measurement model
    Ac = np.array([[0, 1, 0, 0],[0, -0.2, 0, 0],[0,0,0,1],[0,0,0,-0.2]])
    Bc = np.array([[0, 0],[1, 0],[0, 0],[0,1]])
    Cc = np.array([[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]])
    # Cc = np.vstack((Cc,Cc)) # if given enough sensor redundancy
    Dc = np.zeros((Cc.shape[0],Bc.shape[1]))
    s = 3

    # generate discrete-time system
    dtsys_a, dtsys_b, dtsys_c, dtsys_d = SSProblem.convert_ct_to_dt(Ac,Bc,Cc,Dc,TS)
    # define input output data
    init_state1 = np.array([[1.],[1.],[1.],[1.]])
    init_state2 = np.array([[2.],[2.],[2.],[1.]])
    # assume sensors 1,3,5 are under attack
    sensor_initial_states = [init_state2,init_state1,init_state2,init_state1,
                             init_state2,init_state1,init_state1,init_state1]
    u_seq, tilde_y_his, noise_level = SSProblem.generate_attack_measurement(dtsys_a, dtsys_b, dtsys_c, dtsys_d,sensor_initial_states,
                                                                            s = s,is_noisy =  False, noise_level=0.00001)

    # construct a problem instance
    ss_problem = SSProblem(dtsys_a, dtsys_b, dtsys_c, dtsys_d, tilde_y_his, 
                           attack_sensor_count=s,input_sequence=u_seq,measurement_noise_level= noise_level )
    print(f'A: {ss_problem.A},  \n B: {ss_problem.B}, \n C:{ss_problem.C}, \n D:{ss_problem.D}')
    print('input_sequence:',ss_problem.u_seq)
    print('tilde_y_his:',ss_problem.tilde_y_his)

    # define a solution instance
    ssr_solution = SecureStateReconstruct(ss_problem)
    # possible_states,corresp_sensors, _ = ssr_solution.solve(error_bound = 1)
    possible_states,corresp_sensors, _ = ssr_solution.solve_initial_state(error_bound = 0.01)

    if possible_states is not None:
        for ind in range(corresp_sensors.shape[0]):
            sensors = corresp_sensors[ind,:]
            state = possible_states[:,ind]
            print(f'Identified possible states:{state} for sensors {sensors}')

    # test decompostion method
    # y_his = ssr_solution.y_his
    # subprob_a, subprob_c, subprob_y = ssr_solution.gen_subssr_data(y_his)

    # subproblem1 = SSProblem(subprob_a[:,:,0],Bc,Cc,Dc,measurements = subprob_y[:,:,0],input_exist=False)