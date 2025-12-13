import numpy as np
from scipy.special import softmax

eps_, max_ = 1e-12, 1e12
sigmoid = lambda x: 1.0 / (1.0 + clip_exp(-x))
logit   = lambda p: np.log(p) - np.log1p(-p)

def clip_exp(x):
    x = np.clip(x, a_min=-max_, a_max=50)
    y = np.exp(x)
    return np.where(y > 1e-11, y, 0)

class simpleBuffer:
    '''Simple Buffer 2.0
    Update log: 
        To prevent naive writing mistakes,
        we turn the list storage into dict.
    '''
    def __init__(self):
        self.m = {}
        
    def push(self, m_dict):
        self.m = {k: m_dict[k] for k in m_dict.keys()}
        
    def sample(self, *args):
        lst = [self.m[k] for k in args]
        if len(lst)==1: return lst[0]
        else: return lst

class RA():   
    name = 'Random Agent'
    bnds=[(0,1),(0,5)] 
    pbnds= [(.1,.5),(.1,2)]  
    p_name   = ['alpha', 'beta']
    n_params = len(bnds) 

    p_trans = [lambda x: 0.0 + (1 - 0.0) * sigmoid(x),   
               lambda x: 0.0 + (5 - 0.0) * sigmoid(x)]  
    p_links = [lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),  
               lambda y: logit(np.clip((y - 0.0) / (5 - 0.0), eps_, 1 - eps_))]

    def __init__(self,params):
        self._init_mem()
        self._init_critic()
        self._load_params(params)
    
    def _load_params(self, params):
        params = [fn(p) for p, fn in zip(params, self.p_trans)]
        self.alpha = params[0]
        self.beta  = params[1]

    def _init_mem(self):
        self.mem = simpleBuffer()
    def _init_critic(self):
        self.Q = np.array([0.5, 0.5]) 

    # ----------- decision ----------- #
    def policy(self):
        return softmax(self.beta*self.Q)
    def eval_act(self,a):
        '''Evaluate the probability of given state and action
        '''
        prob  = softmax(self.beta*self.Q)
        return prob[int(a)]

    # ----------- learning ----------- #
    def learn(self):
        self.Q = self.Q  
        

class Model1():
    name = 'Model 1'
    bnds = [(0,1),(0,5)] #边界
    pbnds = [(.1,.5),(.1,2)] #采样边界
    p_name   = ['alpha', 'beta']  #参数名
    n_params = len(p_name) 

    p_trans = [lambda x: 0.0 + (1 - 0.0) * sigmoid(x),   
               lambda x: 0.0 + (5 - 0.0) * sigmoid(x)]  
    p_links = [lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),  
               lambda y: logit(np.clip((y - 0.0) / (5 - 0.0), eps_, 1 - eps_))] 
    
    def __init__(self,params):
        self._init_mem()
        self._init_critic()
        self._load_params(params)

    def _init_mem(self):
        self.mem = simpleBuffer()
    def _load_params(self, params):
        params = [fn(p) for p, fn in zip(params, self.p_trans)]
        self.alpha = params[0] # learning rate 
        self.beta  = params[1] # inverse temperature 
    def _init_critic(self):
        self.Q = np.array([0.5, 0.5]) 

    # ----------- decision ----------- #
    def policy(self):
        return softmax(self.beta*self.Q)

    def eval_act(self,a):
        '''Evaluate the probability of given state and action
        '''
        prob  = softmax(self.beta*self.Q)
        return prob[a]
    
        # ----------- learning ----------- #
    def learn(self):
        a, r = self.mem.sample(
                        'a','r')
        
        self.RPE = r - self.Q[a]
        # Q-update
        self.Q[a] = self.Q[a] + self.alpha*self.RPE


class Model2():
    name = 'Model 2'
    bnds = [(0,1), (0,5), (-5,5)]
    pbnds = [(.1,.5), (.1,2), (-3,3)]
    p_name = ['alpha', 'beta', 'kappa_stim']
    n_params = len(p_name)
    
    p_trans = [
        lambda x: 0.0 + (1 - 0.0) * sigmoid(x),
        lambda x: 0.0 + (5 - 0.0) * sigmoid(x),
        lambda x: -5.0 + (10 - 0.0) * sigmoid(x)
    ]
    
    p_links = [
        lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y - 0.0) / (5 - 0.0), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y + 5.0) / (10 - 0.0), eps_, 1 - eps_))
    ]
    
    def __init__(self, params):
        self._init_mem()
        self._init_critic()
        self._load_params(params)

    def _init_mem(self):
        self.mem = simpleBuffer()
    def _load_params(self, params):
        params = [fn(p) for p, fn in zip(params, self.p_trans)]
        self.alpha = params[0]
        self.beta = params[1]
        self.kappa_stim = params[2]
    def _init_critic(self):
        self.Q = np.array([0.5, 0.5])  # [Q_circle, Q_square]
    
    # ----------- decision ----------- #
    def policy(self):
        return softmax(self.beta*self.Q)
    def eval_act(self, a):
        '''Evaluate the probability of given state and action
        '''
        prob  = softmax(self.beta*self.Q)
        return prob[a]
    
    # ----------- learning ----------- #
    def learn(self):
        prev_shape, a, r = self.mem.sample('prev_shape', 'a', 'r')
        
        # Add stimulus stickiness (shape-based)
        if not np.isnan(prev_shape):
            prev_shape = int(prev_shape)
            self.Q[prev_shape] += self.kappa_stim
        
        self.RPE = r - self.Q[a]
        # 更新Q值
        self.Q[a] = self.Q[a] + self.alpha * self.RPE


class Model3():
    name = 'Model 3'
    bnds = [(0,1), (0,1), (0,5)]
    pbnds = [(.1,.5), (.1,.5), (.1,2)]
    p_name = ['alpha_rew', 'alpha_nonrew', 'beta']
    n_params = len(p_name)
    
    p_trans = [
        lambda x: 0.0 + (1 - 0.0) * sigmoid(x),
        lambda x: 0.0 + (1 - 0.0) * sigmoid(x),
        lambda x: 0.0 + (5 - 0.0) * sigmoid(x),
    ]
    
    p_links = [
        lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y - 0.0) / (5 - 0.0), eps_, 1 - eps_)),
    ]
    
    def __init__(self, params):
        self._init_mem()
        self._init_critic()
        self._load_params(params)

    def _init_mem(self):
        self.mem = simpleBuffer()
    def _load_params(self, params):
        params = [fn(p) for p, fn in zip(params, self.p_trans)]
        self.alpha_rew = params[0]
        self.alpha_nonrew = params[1]
        self.beta = params[2]
    def _init_critic(self):
        self.Q = np.array([0.5, 0.5])  # [Q_circle, Q_square]
    
    # ----------- decision ----------- #
    def policy(self):
        return softmax(self.beta*self.Q)
    def eval_act(self, a):
        '''Evaluate the probability of given state and action
        '''
        prob  = softmax(self.beta*self.Q)
        return prob[a]
    
    # ----------- learning ----------- #
    def learn(self):
        a, r = self.mem.sample('a', 'r')
    

        # 选择学习率
        alpha = self.alpha_rew if r > 0 else self.alpha_nonrew
        
        self.RPE = r - self.Q[a]
        # 更新Q值
        self.Q[a] = self.Q[a] + alpha * self.RPE


class Model4():
    name = 'Model 4'
    bnds = [(0,1), (0,1), (0,5), (-5,5)]
    pbnds = [(.1,.5), (.1,.5), (.1,2), (-3,3)]
    p_name = ['alpha_rew', 'alpha_nonrew', 'beta', 'kappa_stim']
    n_params = len(p_name)
    
    p_trans = [
        lambda x: 0.0 + (1 - 0.0) * sigmoid(x),
        lambda x: 0.0 + (1 - 0.0) * sigmoid(x),
        lambda x: 0.0 + (5 - 0.0) * sigmoid(x),
        lambda x: -5.0 + (10 - 0.0) * sigmoid(x)
    ]
    
    p_links = [
        lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y - 0.0) / (5 - 0.0), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y + 5.0) / (10 - 0.0), eps_, 1 - eps_))
    ]
    
    def __init__(self, params):
        self._init_mem()
        self._init_critic()
        self._load_params(params)

    def _init_mem(self):
        self.mem = simpleBuffer()
    def _load_params(self, params):
        params = [fn(p) for p, fn in zip(params, self.p_trans)]
        self.alpha_rew = params[0]
        self.alpha_nonrew = params[1]
        self.beta = params[2]
        self.kappa_stim = params[3]
    def _init_critic(self):
        self.Q = np.array([0.5, 0.5])  # [Q_circle, Q_square]
    
    # ----------- decision ----------- #
    def policy(self):
        return softmax(self.beta*self.Q)
    def eval_act(self, a):
        '''Evaluate the probability of given state and action
        '''
        prob  = softmax(self.beta*self.Q)
        return prob[a]
    
    # ----------- learning ----------- #
    def learn(self):
        prev_shape, a, r = self.mem.sample('prev_shape', 'a', 'r')
        
        # Add stimulus stickiness (shape-based)
        if not np.isnan(prev_shape):
            prev_shape = int(prev_shape)
            self.Q[prev_shape] += self.kappa_stim

        # 选择学习率
        alpha = self.alpha_rew if r > 0 else self.alpha_nonrew
        
        self.RPE = r - self.Q[a]
        # 更新Q值
        self.Q[a] = self.Q[a] + alpha * self.RPE


class Model5():
    name = 'Model 5'
    bnds = [(0,1), (0,1), (0,5), (-5,5)]
    pbnds = [(.1,.5), (.1,.5), (.1,2), (-3,3)]
    p_name = ['alpha_rew', 'alpha_nonrew', 'beta', 'kappa_side']
    n_params = len(p_name)
    
    p_trans = [
        lambda x: 0.0 + (1 - 0.0) * sigmoid(x),
        lambda x: 0.0 + (1 - 0.0) * sigmoid(x),
        lambda x: 0.0 + (5 - 0.0) * sigmoid(x),
        lambda x: -5.0 + (10 - 0.0) * sigmoid(x)
    ]
    
    p_links = [
        lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y - 0.0) / (5 - 0.0), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y + 5.0) / (10 - 0.0), eps_, 1 - eps_))
    ]
    
    def __init__(self, params):
        self._init_mem()
        self._init_critic()
        self._load_params(params)

    def _init_mem(self):
        self.mem = simpleBuffer()
    def _load_params(self, params):
        params = [fn(p) for p, fn in zip(params, self.p_trans)]
        self.alpha_rew = params[0]
        self.alpha_nonrew = params[1]
        self.beta = params[2]
        self.kappa_side = params[3]
    def _init_critic(self):
        self.Q = np.array([0.5, 0.5])  # [Q_circle, Q_square]
    
    # ----------- decision ----------- #
    def policy(self):
        return softmax(self.beta*self.Q)
    def eval_act(self, a):
        '''Evaluate the probability of given state and action
        '''
        prob  = softmax(self.beta*self.Q)
        return prob[a]
    
    # ----------- learning ----------- #
    def learn(self):
        prev_side, circle_side, a, r = self.mem.sample('prev_side', 'circle_side', 'a', 'r')
        
        # Map shape Q-values to spatial Q-values based on current configuration
        if not np.isnan(prev_side):
            prev_side = int(prev_side)
            if circle_side == 1: #circle on left
                self.Q[prev_side] += self.kappa_side  # [Q_left=circle, Q_right=square]
            else:
                self.Q[1-prev_side] += self.kappa_side  # [Q_left=square, Q_right=circle]

        # 选择学习率
        alpha = self.alpha_rew if r > 0 else self.alpha_nonrew
        
        self.RPE = r - self.Q[a]
        # 更新Q值
        self.Q[a] = self.Q[a] + alpha * self.RPE


class Model6():
    name = 'Model 6'
    bnds = [(0,1), (0,1), (0,5), (-5,5), (-5,5)]
    pbnds = [(.1,.5), (.1,.5), (.1,2), (-3,3), (-3,3)]
    p_name = ['alpha_rew', 'alpha_nonrew', 'beta', 'kappa_stim', 'kappa_side']
    n_params = len(p_name)
    
    p_trans = [
        lambda x: 0.0 + (1 - 0.0) * sigmoid(x),
        lambda x: 0.0 + (1 - 0.0) * sigmoid(x),
        lambda x: 0.0 + (5 - 0.0) * sigmoid(x),
        lambda x: -5.0 + (10 - 0.0) * sigmoid(x),
        lambda x: -5.0 + (10 - 0.0) * sigmoid(x)
    ]
    
    p_links = [
        lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y - 0.0) / (5 - 0.0), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y + 5.0) / (10 - 0.0), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y + 5.0) / (10 - 0.0), eps_, 1 - eps_))
    ]
    
    def __init__(self, params):
        self._init_mem()
        self._init_critic()
        self._load_params(params)

    def _init_mem(self):
        self.mem = simpleBuffer()
    def _load_params(self, params):
        params = [fn(p) for p, fn in zip(params, self.p_trans)]
        self.alpha_rew = params[0]
        self.alpha_nonrew = params[1]
        self.beta = params[2]
        self.kappa_stim = params[3]
        self.kappa_side = params[4]
    def _init_critic(self):
        self.Q = np.array([0.5, 0.5])  # [Q_circle, Q_square]
    
    # ----------- decision ----------- #
    def policy(self):
        return softmax(self.beta*self.Q)
    def eval_act(self, a):
        '''Evaluate the probability of given state and action
        '''
        prob  = softmax(self.beta*self.Q)
        return prob[a]
    
    # ----------- learning ----------- #
    def learn(self):
        prev_shape, prev_side, circle_side, a, r = self.mem.sample('prev_shape', 'prev_side', 'circle_side', 'a', 'r')
        
        # Add stimulus stickiness (shape-based)
        if not np.isnan(prev_shape):
            prev_shape = int(prev_shape)
            self.Q[prev_shape] += self.kappa_stim
        
        # Map shape Q-values to spatial Q-values based on current configuration
        if not np.isnan(prev_side):
            prev_side = int(prev_side)
            if circle_side == 1: #circle on left
                self.Q[prev_side] += self.kappa_side  # [Q_left=circle, Q_right=square]
            else:
                self.Q[1-prev_side] += self.kappa_side  # [Q_left=square, Q_right=circle]

        # 选择学习率
        alpha = self.alpha_rew if r > 0 else self.alpha_nonrew
        
        self.RPE = r - self.Q[a]
        # 更新Q值
        self.Q[a] = self.Q[a] + alpha * self.RPE