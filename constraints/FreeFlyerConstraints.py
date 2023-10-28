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

class freeflyer(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu):
        super().__init__(name,ix,iu)
        self.vmax = 0.4
        self.omega_max = np.deg2rad(1)
        self.Tmax = 10*1e-3 #20 * 1e-3
        self.Mmax = 50*1e-6 #100 * 1e-6
        self.av = np.expand_dims(np.zeros(ix),1)
        self.aw = np.expand_dims(np.zeros(ix),1)

    def set_obstacle(self,c,H) :
        self.c = c
        self.H = H
        self.num_obs = len(c)
        
    def forward(self,x,u,xbar,ubar,Q,K,refobs,affine,idx):
        # def forward(self,x,u):
        h = []

        # obstacle avoidance
        if self.num_obs > 0 :
            for obs in refobs :
                h.append(obs[3] + obs[0:2].T@x[0:2]<=obs[2])

        if idx > 0 :
            aQav = affine['aQav'] 
            av = affine['av'] 
            aQa_omega = affine['aQa_omega'] 
            a_omega = affine['a_omega'] 

            # state constraint
            h.append(aQav+av@x <= self.vmax)
            h.append(aQa_omega+a_omega@x <= self.omega_max)
            # h.append(av@x <= self.vmax)
            # h.append(a_omega@x <= self.omega_max)
        aQaT = affine['aQaT'] 
        aT = affine['aT'] 
        aQaM = affine['aQaM'] 
        aM = affine['aM'] 

        # input constraint
        h.append(aQaT+aT@u <= self.Tmax)
        h.append(aQaM+aM@u <= self.Mmax)
        # h.append(aT@u <= self.Tmax)
        # h.append(aM@u <= self.Mmax)

        return h

        


