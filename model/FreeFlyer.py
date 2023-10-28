import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))
from scipy.integrate import solve_ivp
from model.model import OptimalcontrolModel

class freeflyer(OptimalcontrolModel):
    def __init__(self,name,linearzation):
        ix = 12
        iu = 6
        self.iw = 6
        self.iq = 4
        # self.ip = 3
        self.ip = 6
        super().__init__(name,ix,iu,linearzation)
        self.C = np.zeros((self.iq,ix))
        self.C[0,6] = 1
        self.C[1,7] = 1
        # self.C[2,8] = 1
        self.C[2,10] = 1
        self.C[3,11] = 1
        self.D = np.zeros((self.iq,self.iu))
        self.E = np.zeros((ix,self.ip))
        self.E[6,0] = 1
        self.E[6,1] = 1
        self.E[7,2] = 1
        self.E[7,3] = 1
        self.E[8,4] = 1
        self.E[8,5] = 1
        self.G = np.zeros((self.iq,self.iw))
        self.m = 7.2
        self.J = 0.1083

        self.w_in_T = 1*1e-3
        self.w_in_M = 1*1e-6

        
    def forward(self,x,u,idx=None):
        xdim = np.ndim(x)
        if xdim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
        else :
            N = np.size(x,axis = 0)
        udim = np.ndim(u)
        if udim == 1 :
            u = np.expand_dims(u,axis=0)
     
        # state & input
        rx = x[:,0]
        ry = x[:,1]
        rz = x[:,2]

        vx = x[:,3]
        vy = x[:,4]
        vz = x[:,5]

        phi = x[:,6]
        theta = x[:,7]
        psi = x[:,8]

        p = x[:,9]
        q = x[:,10]
        r = x[:,11]

        Tx = u[:,0]
        Ty = u[:,1]
        Tz = u[:,2]
        Mx = u[:,3]
        My = u[:,4]
        Mz = u[:,5]
        
        # output
        f = np.zeros_like(x)
        f[:,0] = vx
        f[:,1] = vy
        f[:,2] = vz

        f[:,3] = 1/self.m * Tx
        f[:,4] = 1/self.m * Ty
        f[:,5] = 1/self.m * Tz

        f[:,6] = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
        f[:,7] = np.cos(phi) * q - np.sin(phi) * r
        f[:,8] = np.sin(phi) / np.cos(theta) * q + np.cos(phi) / np.cos(theta) * r

        # f[:,6] = p
        # f[:,7] = q
        # f[:,8] = r

        f[:,9] = 1/self.J * Mx
        f[:,10] = 1/self.J * My
        f[:,11] = 1/self.J * Mz

        return f

    def forward_uncertain(self,x,u,w,idx=None):
        xdim = np.ndim(x)
        if xdim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            w = np.expand_dims(w,axis=0)
        else :
            N = np.size(x,axis = 0)
     
        # state & input
        rx = x[:,0]
        ry = x[:,1]
        rz = x[:,2]

        vx = x[:,3]
        vy = x[:,4]
        vz = x[:,5]

        phi = x[:,6]
        theta = x[:,7]
        psi = x[:,8]

        p = x[:,9]
        q = x[:,10]
        r = x[:,11]

        Tx = u[:,0]
        Ty = u[:,1]
        Tz = u[:,2]
        Mx = u[:,3]
        My = u[:,4]
        Mz = u[:,5]

        w1 = w[:,0]
        w2 = w[:,1]
        w3 = w[:,2]
        w4 = w[:,3]
        w5 = w[:,4]
        w6 = w[:,5]
        
        # output
        f = np.zeros_like(x)
        f[:,0] = vx
        f[:,1] = vy
        f[:,2] = vz

        f[:,3] = 1/self.m * (Tx + self.w_in_T*w1)
        f[:,4] = 1/self.m * (Ty + self.w_in_T*w2)
        f[:,5] = 1/self.m * (Tz + self.w_in_T*w3)

        f[:,6] = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
        f[:,7] = np.cos(phi) * q - np.sin(phi) * r
        f[:,8] = np.sin(phi) / np.cos(theta) * q + np.cos(phi) / np.cos(theta) * r

        f[:,9] = 1/self.J * (Mx + self.w_in_M*w4)
        f[:,10] = 1/self.J * (My + self.w_in_M*w5)
        f[:,11] = 1/self.J * (Mz + self.w_in_M*w6)

        return f

    def diff_F(self,x,u,w):
        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            w = np.expand_dims(w,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        fw = np.zeros((N,self.ix,6))
        fw[:,3,0] = self.w_in_T/self.m
        fw[:,4,1] = self.w_in_T/self.m
        fw[:,5,2] = self.w_in_T/self.m
        fw[:,9,3] = self.w_in_M/self.J
        fw[:,10,4] = self.w_in_M/self.J
        fw[:,11,5] = self.w_in_M/self.J

        return np.squeeze(fw)
    
    def diff_discrete_zoh_noise(self,x,u,w,delT,tf) :
        # delT = self.delT
        ix = self.ix
        iu = self.iu
        iw = self.iw

        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            w = np.expand_dims(w,axis=0)
        else :
            N = np.size(x,axis = 0)

        def dvdt(t,V,u,w,length) :
            V = V.reshape((length,ix + ix*ix + ix*iu + ix*iw + ix + ix)).transpose()
            x = V[:ix].transpose()
            Phi = V[ix:ix*ix + ix]
            Phi = Phi.transpose().reshape((length,ix,ix))
            Phi_inv = np.linalg.inv(Phi)
            f = self.forward(x,u)
            if self.type_linearization == "numeric_central" :
                A,B = self.diff_numeric_central(x,u)
            elif self.type_linearization == "numeric_forward" :
                A,B = self.diff_numeric(x,u)
            elif self.type_linearization == "analytic" :
                A,B = self.diff(x,u)
            F = self.diff_F(x,u,w)
            dpdt = np.matmul(A,Phi).reshape((length,ix*ix)).transpose()
            dbmdt = np.matmul(Phi_inv,B).reshape((length,ix*iu)).transpose()
            dbpdt = np.matmul(Phi_inv,F).reshape((length,ix*iu)).transpose()
            dsdt = np.squeeze(np.matmul(Phi_inv,np.expand_dims(f,2))).transpose() / tf
            dzdt = np.squeeze(np.matmul(Phi_inv,-np.matmul(A,np.expand_dims(x,2)) - np.matmul(B,np.expand_dims(u,2)))).transpose()
            dv = np.vstack((f.transpose(),dpdt,dbmdt,dbpdt,dsdt,dzdt))
            return dv.flatten(order='F')
        
        A0 = np.eye(ix).flatten()
        B0 = np.zeros((ix*iu))
        F0 = np.zeros((ix*iw))
        s0 = np.zeros(ix)
        z0 = np.zeros(ix)
        V0 = np.array([np.hstack((x[i],A0,B0,F0,s0,z0)) for i in range(N)]).transpose()
        V0_repeat = V0.flatten(order='F')

        sol = solve_ivp(dvdt,(0,delT),V0_repeat,args=(u[0:N],u[1:],N),rtol=1e-6,atol=1e-10)
        # sol = solve_ivp(dvdt,(0,delT),V0_repeat,args=(u[0:N],u[1:],N))
        # IPython.embed()
        idx_state = slice(0,ix)
        idx_A = slice(ix,ix+ix*ix)
        idx_B = slice(ix+ix*ix,ix+ix*ix+ix*iu)
        idx_F = slice(ix+ix*ix+ix*iu,ix+ix*ix+ix*iu+ix*iw)
        idx_s = slice(ix+ix*ix+ix*iu+ix*iw,ix+ix*ix+ix*iu+ix*iw+ix)
        idx_z = slice(ix+ix*ix+ix*iu+ix*iw+ix,ix+ix*ix+ix*iu+ix*iw+ix+ix)
        sol = sol.y[:,-1].reshape((N,-1))
        # xnew = np.zeros((N+1,ix))
        # xnew[0] = x[0]
        # xnew[1:] = sol[:,:ix]
        x_prop = sol[:,idx_state].reshape((-1,ix))
        A = sol[:,idx_A].reshape((-1,ix,ix))
        B = np.matmul(A,sol[:,idx_B].reshape((-1,ix,iu)))
        F = np.matmul(A,sol[:,idx_F].reshape((-1,ix,iu)))
        s = np.matmul(A,sol[:,idx_s].reshape((-1,ix,1))).squeeze()
        z = np.matmul(A,sol[:,idx_z].reshape((-1,ix,1))).squeeze()

        return A,B,F,s,z,x_prop

