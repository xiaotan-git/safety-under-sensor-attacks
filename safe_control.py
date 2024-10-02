import numpy as np
from scipy import linalg
from ss_problem import SSProblem, SecureStateReconstruct
from cvxopt import matrix, solvers


class LinearInequalityConstr():
    '''
    This class defines linear inequality constraints and methods including compare, visualize

    linear inequality constraints: a_mat x + b_vec \geq 0 (or, -a_mat x \leq b_vec)
    a: (M,N) array
    b: (M,1) array 
    '''
    def __init__(self,a:np.ndarray,b:np.ndarray) -> None:
        b = b.reshape(-1,1)
        assert a.shape[0] == b.shape[0]
        self.a_mat = a
        self.b_vec = b

    def compare(self,linear_ineq_constr):
        pass

    def visualize(self,dim = (0,1)):
        pass

    def combine(self,linear_ineq_constr):
        a_mat = np.vstack((self.a_mat, linear_ineq_constr.a_mat))
        b_vec = np.vstack((self.b_vec, linear_ineq_constr.b_vec))
        return LinearInequalityConstr(a_mat,b_vec)
    
    def is_satisfy(self,u):
        u = u.reshape(-1,1)
        assert u.shape[0] == self.a_mat.shape[1]

        return all(self.a_mat@u + self.b_vec +1e-6 >= 0)

class SafeProblem():
    '''
    This class defines data for safe problem
    '''
    def __init__(self,ss_problem:SSProblem,h,q,gamma) -> None:
        
        assert gamma>=0
        assert gamma<=1

        assert h.shape[0] == q.shape[0]
        assert h.shape[1] == ss_problem.A.shape[1]
        assert q.shape[1] == 1

        self.problem = ss_problem

        self.h = h
        self.q = q
        self.gamma = gamma

        # according to (7) of the note "Safety of linear systems under sensor attacks without state estimation
        self.k = (1-gamma)*h@np.linalg.matrix_power(self.problem.A,self.problem.io_length) - h@np.linalg.matrix_power(self.problem.A,self.problem.io_length + 1)

    def cal_cbf_condition_state(self,state)-> LinearInequalityConstr:
        state = state.reshape(-1,1)
        a_mat = self.h @ self.problem.B
        b_vec = self.h @ self.problem.A @ state + self.q - (1-self.gamma)*(self.h@state + self.q)
        return LinearInequalityConstr(a_mat, b_vec)
    
    def merge_multiple_LIC(self,constr_list)-> LinearInequalityConstr:
        lic_full = constr_list[0]
        if len(constr_list) == 1:
            return lic_full
        
        for lic in constr_list[1:]:
            lic_full = LinearInequalityConstr.combine(lic_full,lic)
        return lic_full

    def cal_cbf_condition(self, possible_states) -> LinearInequalityConstr:
        '''
        This method calculates CBF conditions over a set of states
        '''
        cbf_conditions = []
        for state in possible_states:
            cbf_condition = self.cal_cbf_condition_state(state)
            cbf_conditions.append(cbf_condition)

        total_cbf_condition = self.merge_multiple_LIC(cbf_conditions)
        return total_cbf_condition

    def cal_Q_ut(self):
        '''
        construct Q(u(t)) according to (7) of the note "Safety of linear systems under sensor attacks without state estimation

        '''
        
        H = self.h
        q = self.q
        gamma = self.gamma

        A = self.problem.A
        B = self.problem.B
        io_length = self.problem.io_length
        n = self.problem.n
        m = self.problem.m

        u_seq = self.problem.u_seq

        left_mat = H@((1-gamma)*np.identity(n)- A)
        right_vec = np.zeros((n,1))
        for k in range(io_length-1):
            ut = u_seq[k,:]
            ut = ut.reshape(-1,1)
            temp = linalg.fractional_matrix_power(A,io_length-2-k)@B@ut # note in the paper, io_length = t + 1
            right_vec = right_vec+ temp
        
        return left_mat@right_vec - gamma*q

    def cal_safe_input_constr_woSSR(self,initial_states_subssr)-> LinearInequalityConstr:
        '''
        This method implements our computationally efficient CBF conditions
        initial_states_subssr is a list of possible_initial_states, each entry correspnds to one subspace
        possible_initial_states = initial_states_subssr[j] is a n*x 2d array, each column corresponds to one state
        
        '''
        Q_ut = self.cal_Q_ut()
        kv_maxsum = np.zeros(np.shape(self.q))
        count_subspace = len(initial_states_subssr)
        for j in range(count_subspace):
            possible_initial_states = initial_states_subssr[j]
            kv = self.k @ possible_initial_states
            kv_max = kv.max(axis = 1)
            kv_max = kv_max.reshape(-1,1)
            kv_maxsum = kv_maxsum + kv_max 
        # a_mat x + b_vec \geq 0
        a_mat = self.h @ self.problem.B
        b_vec = - kv_maxsum - Q_ut

        # print(f'a_mat: {a_mat.shape}')
        # print(f'b_vec: {b_vec.shape}')
        # print(f'kv_maxsum: {kv_maxsum.shape}')
        # print(f'Q_ut shape: {Q_ut}')
        return LinearInequalityConstr(a_mat,b_vec)

    def cal_safe_qp(self,u_nom, lic:LinearInequalityConstr):
        '''
        sol_flag: -1 -- qp solver fails, 0 -- qp status "unknown", 1 -- qp status 'optimal'
        '''
        flag = 0
        # Define QP parameters for qp solver (CVXOPT): 
        # min 0.5 xT P x+ qT x s.t. Gx <= h
        qp_P = matrix(np.identity(u_nom.shape[0]))
        qp_q = matrix(-u_nom,(u_nom.shape[0],1),'d') #  qp_q is a m*1 matrix
        qp_G = matrix(-lic.a_mat) 
        qp_h = matrix(lic.b_vec,(lic.b_vec.shape[0],1),'d') # qp_h is a x*1 matrix

        solvers.options['show_progress'] = False # mute optimization output
        solvers.options['maxiters'] = 500 # increase max iteration number
        solvers.options['abstol'] = 1e-4
        solvers.options['reltol'] = 1e-5
        try:
            solv_sol = solvers.qp(qp_P,qp_q,qp_G,qp_h)
            if solv_sol['status'] == 'unknown':
                print('warning: safe control is approximately computed. Use u_nom instead.')
                u = u_nom.flatten()
            else:
                flag = 1
                u = np.array(solv_sol['x']).flatten() # extract decision variables of optimization problem
        except:
            # print('cvxopt solver failed:')
            # print(f'qp_P: {qp_P}')
            # print(f'qp_q: {qp_q}')
            # print(f'qp_G: {qp_G}')
            # print(f'qp_h: {qp_h}')

            # solvers.options['show_progress'] = True
            # solv_sol = solvers.qp(qp_P,qp_q,qp_G,qp_h)

            # raise TypeError('no control input found')
            print('warning: safe control fails. Use u_nom instead.')
            flag = -1
            u = u_nom.flatten()
        

        return u,flag
    
    def cal_safe_control(self,u_nom, possible_states):
        lic = self.cal_cbf_condition(possible_states)
        # print(f'lic.a_mat:\n {lic.a_mat}')
        # print(f'lic.b_vec:\n {lic.b_vec}')
        u,flag = self.cal_safe_qp(u_nom,lic)
        return u,lic,flag

    def max_kv(self):
        possible_init_states,_,_ = self.ssr.solve_initial_state()
        kv = self.k @ possible_init_states
        kv_max = kv.max(axis = 1)
        return kv_max
    
    def max_test_formula(self):
        vec_list = []
        for comb in self.ssr.possible_comb:
            obser_matrix = self.ssr.construct_obser_matrix(comb)
            measure_vec = self.ssr.construct_measurement_his(comb)
            vec = np.linalg.pinv(obser_matrix)@measure_vec
            vec_list.append(vec)
        vecs = np.hstack(vec_list)
        tf = self.k@vecs
        tf_max = tf.max(axis = 1)
        return tf_max
    
    def test2(self):
        diff_list = []
        for comb in self.ssr.possible_comb:
            obser_matrix = self.ssr.construct_obser_matrix(comb)
            measure_vec = self.ssr.construct_measurement_his(comb)
            lhs = self.k @ np.linalg.pinv(obser_matrix)@measure_vec
            
            for sensor_i in range(self.problem.p):
                if sensor_i not in comb:
                    # Create a new combination by copying and extending the original combination
                    comb_ext = comb.copy()
                    comb_ext.append(sensor_i)
                    obser_matrix_ext = self.ssr.construct_obser_matrix(comb_ext)
                    measure_vec_ext = self.ssr.construct_measurement_his(comb_ext)
                    rhs = self.k @ np.linalg.pinv(obser_matrix_ext)@measure_vec_ext

                    diff = lhs - rhs
                    diff_list.append(diff)
        
        diff_array = np.array(diff_list)
        return diff_array


if __name__ =='__main__':
    pass
    # # define system model and measurement model
    # Ac = np.array([[0, 1, 0, 0],[0, -0.2, 0, 0],[0,0,0,1],[0,0,0,-0.2]])
    # Bc = np.array([[0, 0],[1, 0],[0, 0],[0,1]])
    # Cc = np.array([[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]])
    # # Cc = np.vstack((Cc,Cc)) # if given enough sensor redundancy
    # Dc = np.zeros((Cc.shape[0],Bc.shape[1]))
    # p = np.shape(Cc)[0]
    # s = 3
    # ss_problem = SSProblem(Ac,Bc,Cc,Dc,s)

    # # define attacking model
    # fake_state_count = 1
    # init_states = np.array([[1.,2.],[1.,2.],[1.,2.],[1.,1.]])
    # att_ind = [0,2,4] # sensors 1, 3, 5
    # # att_dic: which initial state and its corresponding sensors
    # # 0 -- healthy sensor, 1, 2, ... -- attacked sensors
    # att_dic = {0:[i for i in range(p) if i not in att_ind], 1:att_ind}

    # ss_problem.gen_attack_measurement(s=3,att_dic = att_dic,fake_state_count=fake_state_count,
    #                                     init_states=init_states,noise=True)
    
    # h = np.array([[1,0,0,0],[-1,0,0,0],[0,1,0,0],[0,-1,0,0],[0,0,1,0],[0,0,-1,0],[0,0,0,1],[0,0,0,-1]])
    # q = np.array([[4],[-4],[4],[-4],[4],[-4],[4],[-4]])
    # gamma = 0.5
    # safe_prob = SafeProblem(ss_problem,h,q,gamma)

    # # test for Question 1
    # print(f'max Kv: {safe_prob.max_kv()}')
    # print(f'max test formula: {safe_prob.max_test_formula()}')
    # diff = safe_prob.max_kv() - safe_prob.max_test_formula()
    # print(f'difference: {diff}, max diff: {diff.max()}')
    
    # # test for Question 2
    # diff = safe_prob.test2()
    # diff_min = diff.min(axis=1)
    # diff_max = diff.max(axis=1)
    # diff_mmin = diff.min()
    # diff_mmax = diff.max()
    # print(f'diff min: {diff_min}, \n max: {diff_max}, \n min min: {diff_mmin}, \n max max:  {diff_mmax}')