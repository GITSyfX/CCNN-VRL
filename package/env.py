import numpy as np

class two_stage:
    nS = 9 #1stage(inital)+4stage(step1)+4rewardstage(stage2)
    nA = 2 #left/right
    nT = 3
    s_termination = list(range(5,9))

    def __init__(self,seed=1234):
        '''A MDP is a 3-element-tuple

        S: state space
        A: action space
        T: transition function
        R: reward
        '''
        self.rng = np.random.RandomState(seed)
        self.nS = two_stage.nS
        self.nA = two_stage.nA
        self._init_state()
        self._init_action()
        self._init_level()
        self._init_map()
        self._init_trans()
        self._init_goal()
        self._init_reward()

    def _init_state(self):
        self.S  = np.arange(self.nS)
    
    def _init_action(self):
        self.A  = np.arange(self.nA)  

    def _init_level(self):
        self.level = np.array([1,2,2,2,2,3,3,3,3])

    def _init_map(self):
        self.T = {0:[[1,2], [6,7], [7,8], [6,5], [6,8]],
                  1:[[3,4], [7,8], [6,8], [5,8], [8,5]]}

    def _init_trans(self):
        self.A_prob = {
                0:np.zeros((self.nS,self.nS),dtype=float),
                1:np.zeros((self.nS,self.nS),dtype=float)
                } 
        
    def _init_goal(self):
        self._last_g = None

    def _init_reward(self):
        '''R(r|s)''' 
        self.R = np.zeros(self.nS)
        self.R_save = np.array([0,0,0,0,0,40,20,10,0])
        self.R_prob = np.zeros(self.nS)
        self.R_prob[5:self.nS] = np.array([1,1,1,1]) # reward delivery prob
    
    def set_reward(self, g):
        if self._last_g != g:
            self.R.fill(0)
            if g == -1:
                self.R[5:self.nS] = np.array([40, 20, 10, 0])
            else:
                self.R[g] = self.R_save[g]
            self._last_g = g

    def reset(self):
        '''always start with state=0
        '''
        self.s = 0
        self.done = False 
        return self.s 
    
    def step(self,s,a,g,p):
        for mm in range(self.nA):
            for nn in range(len(self.T[mm])):
                self.A_prob[mm][nn,self.T[mm][nn]]=[p,1-p]
        state_prob = self.A_prob[a][s]

        s_next = self.rng.choice(self.S,p=state_prob)
        # get the reward
        self.set_reward(g)
        r = self.R[s_next]

        # check the termination
        if s_next in two_stage.s_termination: self.done = True 
        else: self.done = False
    
        # move on 
        self.s = s_next 

        return self.s, r, self.done 