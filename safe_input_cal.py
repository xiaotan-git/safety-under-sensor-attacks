import numpy as np
from ss_problem import SSProblem, SecureStateReconst

class LinearInequalityConstr():
    '''
    This class defines linear inequality constraints and methods including compare, visualize

    linear inequality constraints are given in forms of ax >= b
    a: (M,N) array
    b: (M,1) array 
    '''
    def __init__(self,a,b) -> None:
        self.a_mat = a
        self.b_vec = b

    def compare(self,linear_ineq_constr):
        pass

    def visualize(self,dim = (0,1)):
        pass

class SafeProblemUnderAttack():
    '''
    This class defines data for safe problem
    '''
    def __init__(self,ss_problem:SSProblem,h,q,gamma) -> None:
        self.problem = ss_problem
        self.ssr = SecureStateReconst(ss_problem)

        self.a = ss_problem.A
        self.b = ss_problem.B
        self.c = ss_problem.C
        self.d = ss_problem.D
        self.u_seq = ss_problem.u_seq

        self.h = h
        self.q = q
        self.gamma = gamma
        self.tspan = ss_problem.tspan

        self.hb = h@self.b
        self.k = (1-gamma)*h@np.linalg.matrix_power(self.a,self.tspan) - h@np.linalg.matrix_power(self.a,self.tspan + 1)


    def cal_safe_input_constr_wSSR(self, states) -> LinearInequalityConstr:
        '''
        This method calculates CBF conditions over a set of states
        '''
        pass

    def cal_safe_input_constr_woSSR(self,u_seq,y_his)-> LinearInequalityConstr:
        '''
        This method implements CBF conditions over a set of input-output data
        '''
        pass

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
    
    h = np.array([[1,0,0,0],[-1,0,0,0],[0,1,0,0],[0,-1,0,0],[0,0,1,0],[0,0,-1,0],[0,0,0,1],[0,0,0,-1]])
    q = np.array([[4],[-4],[4],[-4],[4],[-4],[4],[-4]])
    gamma = 0.5
    safe_prob = SafeProblemUnderAttack(ss_problem,h,q,gamma)

    # test for Question 1
    print(f'max Kv: {safe_prob.max_kv()}')
    print(f'max test formula: {safe_prob.max_test_formula()}')
    diff = safe_prob.max_kv() - safe_prob.max_test_formula()
    print(f'difference: {diff}, max diff: {diff.max()}')
    
    # test for Question 2
    diff = safe_prob.test2()
    diff_min = diff.min(axis=1)
    diff_max = diff.max(axis=1)
    diff_mmin = diff.min()
    diff_mmax = diff.max()
    print(f'diff min: {diff_min}, \n max: {diff_max}, \n min min: {diff_mmin}, \n max max:  {diff_mmax}')