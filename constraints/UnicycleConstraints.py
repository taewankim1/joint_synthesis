import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
import cvxpy as cvx
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))


from constraints.constraints import OptimalcontrolConstraints

def get_obs_ab(c,H,xbar) :
    hr = 1 - cvx.norm(H@(xbar[0:2]-c))
    dhdr = - (H.T@H@(xbar[0:2]-c)/cvx.norm(H@(xbar[0:2]-c))).T
    a = dhdr
    b = dhdr@xbar[0:2] - hr
    return  a,b

class UnicycleConstraints(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu):
        super().__init__(name,ix,iu)
        self.idx_bc_f = slice(0, ix)
        self.ih = 4

        self.vmax = 3.0
        self.vmin = 0.0

        self.wmax = 2.5
        self.wmin = -2.5


    def set_obstacle(self,c,H) :
        self.c = c
        self.H = H
        
    def forward(self,x,u,xbar,ubar,Q,K,refobs,aQav,aQaw):
        h = []
        # obstacle avoidance
        # def get_obs_const(c1,H1) :
        #     a,b = get_obs_ab(c1,H1,xbar)
        #     h_Q = cvx.sqrt(a.T@Q[0:2,0:2]@a)
        #     return h_Q+a.T@x[0:2] <= b
        # if self.H is not None :
        #     for c1,H1 in zip(self.c,self.H) :
        #         h.append(get_obs_const(c1,H1))

        for obs in refobs :
            h.append(obs[3] + obs[0:2].T@x[0:2]<=obs[2])

        # input constraints
        a = np.expand_dims(np.array([1,0]),1)
        h.append(aQav + a.T@u <= self.vmax)
        h.append(aQav - a.T@u <= -self.vmin)

        a = np.expand_dims(np.array([0,1]),1)
        h.append(aQaw + a.T@u <= self.wmax)
        h.append(aQaw - a.T@u <= -self.wmin)
        return h

    def forward_bf(self,x,u,xbar,ubar,Q,K,bf):
        h = []
        idx_bf = 0
        # obstacle avoidance
        def get_obs_const(c1,H1,bf_) :
            a,b = get_obs_ab(c1,H1,xbar)
            h_Q = np.sqrt(a.T@Q[0:2,0:2]@a)
            # print(Q)
            return h_Q+a.T@x[0:2] + bf_ <= b
            # return 1 - np.linalg.norm(H1@(xbar[0:2]-c1)) - (H1.T@H1@(xbar[0:2]-c1)/np.linalg.norm(H1@(xbar[0:2]-c1))).T@(x[0:2]-xbar[0:2]) <= 0
        if self.H is not None :
            for c1,H1 in zip(self.c,self.H) :
                h.append(get_obs_const(c1,H1,bf[idx_bf]))
                idx_bf+=1
        # input constraints
        a = np.expand_dims(np.array([1,0]),1)
        h.append(np.sqrt(a.T@K@Q@K.T@a) + a.T@u + bf[idx_bf] <= self.vmax)
        idx_bf+=1
        h.append(np.sqrt(a.T@K@Q@K.T@a) - a.T@u + bf[idx_bf] <= -self.vmin)
        idx_bf+=1

        a = np.expand_dims(np.array([0,1]),1)
        h.append(np.sqrt(a.T@K@Q@K.T@a) + a.T@u + bf[idx_bf] <= self.wmax)
        idx_bf+=1
        h.append(np.sqrt(a.T@K@Q@K.T@a) - a.T@u + bf[idx_bf]<= -self.wmin)
        return h

        


