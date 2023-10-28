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
    def __init__(self,name,ix,iu,vmax=3,vmin=0,wmax=2.5,wmin=-2.5):
        super().__init__(name,ix,iu)
        self.idx_bc_f = slice(0, ix)
        self.ih = 4

        self.vmax = vmax
        self.vmin = vmin

        self.wmax = wmax
        self.wmin = wmin

        self.av = np.expand_dims(np.array([1,0]),1)
        self.aw = np.expand_dims(np.array([0,1]),1)



    def set_obstacle(self,c,H) :
        self.c = c
        self.H = H
        
    def forward(self,x,u,xbar,ubar,Q,K,refobs,affine,idx=None):
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

        aQav = affine['aQav']
        aQaw = affine['aQaw']

        # input constraints
        a = np.expand_dims(np.array([1,0]),1)
        h.append(aQav + a.T@u <= self.vmax)
        h.append(aQav - a.T@u <= -self.vmin)

        a = np.expand_dims(np.array([0,1]),1)
        h.append(aQaw + a.T@u <= self.wmax)
        h.append(aQaw - a.T@u <= -self.wmin)
        return h

    def get_const_state(self,xnom,unom) :
        c_list,H_list = self.c,self.H

        const_state = []
        c,H = c_list[0],H_list[0] # temporary
        M = np.array([[1,0,0],[0,1,0]])
        N = len(xnom) 
        for c,H in zip(c_list,H_list) :
            tmp_zip = {}
            a = np.zeros((N,3,1))
            bb = np.zeros(N)
            for i in range(N) :
                x = xnom[i]
                deriv  = - M.T@H.T@H@(M@x-c) / np.linalg.norm(H@(M@x-c))
                s = 1 - np.linalg.norm(H@(M@x-c))
                a[i,:,0] = deriv
                b = -s + deriv@x
                bb[i] = (b - a[i,:,0].T@x) ** 2
            tmp_zip['a'] = np.squeeze(a)
            tmp_zip['(b-ax)^2'] = np.squeeze(bb)
            const_state.append(tmp_zip)
        return const_state


    def get_const_input(self,xnom,unom) :
        N = len(unom)
        const_input = []
        a = np.zeros((N,2,1))
        a[:,0,:] = 1
        b = self.vmax * np.ones(N)
        c = (b - np.squeeze(np.transpose(a,(0,2,1))@unom[:,:,np.newaxis])) ** 2
        tmp_zip = {}
        tmp_zip['a'] = np.squeeze(a)
        tmp_zip['(b-au)^2'] = c
        const_input.append(tmp_zip)

        a = np.zeros((N,2,1))
        a[:,0,:] = -1
        b = -self.vmin * np.ones(N)
        c = (b - np.squeeze(np.transpose(a,(0,2,1))@unom[:,:,np.newaxis])) ** 2
        tmp_zip = {}
        tmp_zip['a'] = np.squeeze(a)
        tmp_zip['(b-au)^2'] = c
        const_input.append(tmp_zip)
        
        a = np.zeros((N,2,1))
        a[:,1,:] = 1
        b = self.wmax * np.ones(N)
        c = (b - np.squeeze(np.transpose(a,(0,2,1))@unom[:,:,np.newaxis])) ** 2
        tmp_zip = {}
        tmp_zip['a'] = np.squeeze(a)
        tmp_zip['(b-au)^2'] = c
        const_input.append(tmp_zip)

        a = np.zeros((N,2,1))
        a[:,1,:] = -1
        b = -self.wmin * np.ones(N)
        c = (b - np.squeeze(np.transpose(a,(0,2,1))@unom[:,:,np.newaxis])) ** 2
        tmp_zip = {}
        tmp_zip['a'] = np.squeeze(a)
        tmp_zip['(b-au)^2'] = c
        const_input.append(tmp_zip)

        return const_input

        


