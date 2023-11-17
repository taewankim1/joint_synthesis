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

    def get_const_state(self,xnom,unom) :
        c_list,H_list = self.c,self.H

        const_state = []
        M = np.zeros((2,self.ix))
        M[0,0] = 1
        M[1,0] = 1
        N = len(xnom) 
        # # obstacle avoidance
        # for c,H in zip(c_list,H_list) :
        #     tmp_zip = {}
        #     a = np.zeros((N,self.ix,1))
        #     bb = np.zeros(N)
        #     for i in range(N) :
        #         x = xnom[i]
        #         deriv  = - M.T@H.T@H@(M@x-c) / np.linalg.norm(H@(M@x-c))
        #         s = 1 - np.linalg.norm(H@(M@x-c))
        #         a[i,:,0] = deriv
        #         b = -s + deriv@x
        #         bb[i] = (b - a[i,:,0].T@x) ** 2
        #     tmp_zip['a'] = np.squeeze(a)
        #     tmp_zip['(b-ax)^2'] = np.squeeze(bb)
        #     const_state.append(tmp_zip)

        # state constraints
        tmp_zip = {}
        a = np.zeros((N,self.ix,1))
        for i in range(N) :
            x = xnom[i]
            A = np.vstack((np.zeros((3,3)),np.eye(3),np.zeros((6,3)))).T
            a[i,:,0] = x[:,np.newaxis].T@A.T@A / np.linalg.norm(A@x[:,np.newaxis])
        b = self.vmax * np.ones(N)
        c = (b - np.squeeze(np.transpose(a,(0,2,1))@xnom[:,:,np.newaxis])) ** 2
        tmp_zip['a'] = np.squeeze(a)
        tmp_zip['(b-ax)^2'] = c
        const_state.append(tmp_zip)

        tmp_zip = {}
        a = np.zeros((N,self.ix,1))
        for i in range(N) :
            x = xnom[i]
            A = np.vstack((np.zeros((9,3)),np.eye(3))).T
            a[i,:,0] = x[:,np.newaxis].T@A.T@A / np.linalg.norm(A@x[:,np.newaxis])
        b = self.omega_max * np.ones(N)
        c = (b - np.squeeze(np.transpose(a,(0,2,1))@xnom[:,:,np.newaxis])) ** 2
        tmp_zip['a'] = np.squeeze(a)
        tmp_zip['(b-ax)^2'] = c
        const_state.append(tmp_zip)
        return const_state


    def get_const_input(self,xnom,unom) :
        N = len(unom)
        const_input = []
        a = np.zeros((N,self.iu,1))
        for i in range(N) :
            u = unom[i]
            A = np.vstack((np.eye(3),np.zeros((3,3)))).T
            a[i,:,0] = u[:,np.newaxis].T@A.T@A / np.linalg.norm(A@u[:,np.newaxis])
        b = self.Tmax * np.ones(N)
        c = (b - np.squeeze(np.transpose(a,(0,2,1))@unom[:,:,np.newaxis])) ** 2
        tmp_zip = {}
        tmp_zip['a'] = np.squeeze(a)
        tmp_zip['(b-au)^2'] = c
        const_input.append(tmp_zip)

        a = np.zeros((N,self.iu,1))
        for i in range(N) :
            u = unom[i]
            A = np.vstack((np.zeros((3,3)),np.eye(3))).T
            a[i,:,0] = u[:,np.newaxis].T@A.T@A / np.linalg.norm(A@u[:,np.newaxis])
        b = self.Mmax * np.ones(N)
        c = (b - np.squeeze(np.transpose(a,(0,2,1))@unom[:,:,np.newaxis])) ** 2
        tmp_zip = {}
        tmp_zip['a'] = np.squeeze(a)
        tmp_zip['(b-au)^2'] = c
        const_input.append(tmp_zip)
        return const_input   


