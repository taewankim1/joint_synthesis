import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np
import cvxpy as cvx
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))
from utils.utils_alg import get_sample_eta_w
import IPython

class Lipschitz :
    def __init__(self,name,ix,iu,iq,ip,iw,N,num_sample=100,flag_uniform=True) :
        self.ix = ix
        self.iu = iu
        self.iq = iq
        self.ip = ip
        self.iw = iw
        self.N = N
        self.name = name

        self.num_sample = num_sample
        zs_sample = [] # sample in unit sphere will be projected to ellipse (Q_k)
        zw_sample = [] 
        self.flag_uniform = flag_uniform
        if flag_uniform == True and 'unicycle' in self.name :
            assert num_sample == 100, f"number of samples should be 100 for the uniform case"
            # uniformly fixed - unicycle
            for i in np.linspace(-1.0, 1.0, num=5) :
                for j in np.linspace(-1.0, 1.0, num=5) :
                    for k in np.linspace(-1.0, 1.0, num=4) :
                        z = np.array([i,j,k])
                        zs = z / np.linalg.norm(z)
                        zs_sample.append(zs)
            for i in np.linspace(-1.0, 1.0, num=10) :
                for j in np.linspace(-1.0, 1.0, num=10) :
                    z = np.array([i,j])
                    zw = z / np.linalg.norm(z)
                    zw_sample.append(zw)
        elif flag_uniform == True and 'freeflyer' in self.name :
            assert num_sample == 256
            for i in np.linspace(-1.0, 1.0, num=4) :
                for j in np.linspace(-1.0, 1.0, num=4) :
                    for k in np.linspace(-1.0, 1.0, num=4) :
                        for l in np.linspace(-1.0, 1.0, num=4) :
                            z = np.zeros(ix)
                            z[7] = i
                            z[8] = j
                            z[10] = k
                            z[11] = l
                            zs = z / np.linalg.norm(z)
                            zs_sample.append(zs)
            for _ in range(self.num_sample) :
                # z = np.random.randn(iw)
                # zw = z / np.linalg.norm(z)
                zw = np.zeros(iw)
                zw_sample.append(zw)
        else :
            for _ in range(self.num_sample) :
                z = np.random.randn(ix)
                zs = z / np.linalg.norm(z)
                zs_sample.append(zs)
                
                z = np.random.randn(iw)
                zw = z / np.linalg.norm(z)
                zw_sample.append(zw)
        self.zs_sample = zs_sample
        self.zw_sample = zw_sample

    def initialize(self,xbar,ubar,xprop,Qbar,Kbar,A,B,C,D,E,F,G,myModel)  :
        ix,iu,N = self.ix,self.iu,self.N
        ip,iq,iw = self.ip,self.iq,self.iw

        self.xbar = xbar
        self.ubar = ubar
        self.xprop = xprop
        self.Qbar = Qbar
        self.Kbar = Kbar

        self.A = A
        self.B = B 
        self.C = C 
        self.D = D 
        self.E = E 
        self.F = F 
        self.G = G

        self.myModel = myModel

    def update_lipschitz_norm(self,myModel,delT) :
        ix,iu,N = self.ix,self.iu,self.N
        ip,iq,iw = self.ip,self.iq,self.iw
        A,B,C, = self.A,self.B,self.C
        D,E,F,G = self.D,self.E,self.F,self.G
        Qbar,Kbar = self.Qbar,self.Kbar
        xbar,ubar = self.xbar,self.ubar
        xprop = self.xprop


        gamma = np.zeros((N))
        for idx in range(N) :
            eta_sample,w_sample = get_sample_eta_w(Qbar[idx],self.zs_sample,self.zw_sample) 
            num_sample = len(self.zs_sample)
            gamma_val = []
            for idx_s in range(num_sample) :
                es,ws = eta_sample[idx_s],w_sample[idx_s] # samples on surface of ellipse

                xii = Kbar[idx]@es

                xs_prop = propagate_model(myModel,xbar[idx]+es,ubar[idx]+xii,ws,delT)
                eta_prop = xs_prop - xbar[idx+1]

                LHS = eta_prop - (A[idx]+B[idx]@Kbar[idx])@es - F[idx]@ws + xbar[idx+1] - xprop[idx+1]
                # if idx == 0 and idx_s == 0 :
                #     IPython.embed()
                # print(LHS)
                if LHS[2] > 1e-6 :
                    print("LHS[2] > 1e-6: is this possible?")

                mu = (C+D@Kbar[idx])@es + G@ws
                # print(LHS[6:9])
                # print(np.linalg.norm(LHS))
                # print(mu)
                gamma_val.append(np.linalg.norm(LHS)/np.linalg.norm(mu))
            gamma_val = np.array(gamma_val)

            gamma[idx] = np.max(gamma_val)
            # print("gamma[{:}]".format(idx),gamma[idx])

        return gamma

    # def update_lipschitz(self,myModel,delT) :
    #     ix,iu,N = self.ix,self.iu,self.N
    #     ip,iq,iw = self.ip,self.iq,self.iw
    #     A,B,C, = self.A,self.B,self.C
    #     D,E,F,G = self.D,self.E,self.F,self.G
    #     Qbar,Kbar = self.Qbar,self.Kbar
    #     xbar,ubar = self.xbar,self.ubar
    #     xprop = self.xprop

    #     gamma = np.zeros((N))
    #     for idx in range(N) :
    #         eta_sample,w_sample = get_sample_eta_w(Qbar[idx],self.zs_sample,self.zw_sample) 
    #         num_sample = len(self.zs_sample)

    #         constraints = []
    #         Delta_list = []
    #         gamma_val = []
    #         max_val = 0

    #         xnew_prop = xprop[idx]
    #         for idx_s in range(num_sample) :
    #             Delta = cvx.Variable((ip,iq))
    #             nu = cvx.Variable(ix)
    #             es,ws = eta_sample[idx_s],w_sample[idx_s] # samples on surface of ellipse
    #             # temporary
    #             ws *= 0

    #             xii = Kbar[idx]@es

    #             xs_prop = propagate_model(myModel,xbar[idx]+es,ubar[idx]+xii,delT)
    #             eta_prop = xs_prop - xbar[idx+1]

    #             LHS = eta_prop - (A[idx]+B[idx]@Kbar[idx])@es - F[idx]@ws + xbar[idx+1] - xnew_prop
    #             if LHS[2] > 1e-6 :
    #                 print("LHS[2] > 1e-6: is this possible?")

    #             mu = (C+D@Kbar[idx])@es + G@ws

    #             constraints.append(LHS == E@Delta@mu + 0*nu)
    #             cost = cvx.norm(Delta,'fro') + 1e4*cvx.norm(nu)
    #             prob = cvx.Problem(cvx.Minimize(cost),constraints)
                
    #             prob.solve()
    #             if prob.status != "optimal" :
    #                 print(idx_s,self.prob.status)
    #             if prob.value >= max_val :
    #                 idx_max = idx_s
    #                 max_val = prob.value 
    #             Delta_list.append(Delta.value)
    #             gamma_val.append(cvx.norm(Delta,2).value)
    #         Delta_list = np.array(Delta_list)
    #         gamma_val = np.array(gamma_val)

    #         gamma[idx] = np.max(gamma_val)
    #         print("gamma[{:}]".format(idx),gamma[idx],"nu",cvx.norm(nu).value)

    #     return gamma
    
    def update_lipschitz_parallel(self,myModel,delT) :
        ix,iu,N = self.ix,self.iu,self.N
        ip,iq,iw = self.ip,self.iq,self.iw
        A,B,C, = self.A,self.B,self.C
        D,E,F,G = self.D,self.E,self.F,self.G
        Qbar,Kbar = self.Qbar,self.Kbar
        xbar,ubar = self.xbar,self.ubar
        xprop = self.xprop

        gamma = np.zeros((N))
        for idx in range(N) :
            eta_sample,w_sample = get_sample_eta_w(Qbar[idx],self.zs_sample,self.zw_sample) 
            num_sample = len(self.zs_sample)

            Delta = cvx.Variable((num_sample,ip*iq))
            nu = cvx.Variable((num_sample,ix))

            constraints = []
            cost_list = []
            # deltanorm_list = []

            # xnew_prop = xprop[idx]
            for idx_s in range(num_sample) :
                es,ws = eta_sample[idx_s],w_sample[idx_s] # samples on surface of ellipse

                xii = Kbar[idx]@es

                xs_prop = propagate_model(myModel,xbar[idx]+es,ubar[idx]+xii,ws,delT)
                eta_prop = xs_prop - xbar[idx+1]

                LHS = eta_prop - (A[idx]+B[idx]@Kbar[idx])@es - F[idx]@ws + xbar[idx+1] - xprop[idx+1]
                # LHS = eta_prop - (A[idx]+B[idx]@Kbar[idx])@es - F[idx]@ws + xbar[idx+1] - xnew_prop
                if LHS[2] > 1e-6 :
                    print("LHS[2] > 1e-6: is this possible?")

                mu = (C+D@Kbar[idx])@es + G@ws
                delta = cvx.reshape(Delta[idx_s],(ip,iq))

                constraints.append(LHS == E@delta@mu + 0*nu[idx_s])
                # cost_list.append(cvx.norm(delta,2))
                cost_list.append(cvx.norm(delta,'fro'))
                # deltanorm_list.append(cvx.norm(delta,2))

            cost = cvx.norm(Delta,'fro') + 1e4*cvx.norm(nu,'fro')
            # cost = cvx.norm(Delta,2) + 1e4*cvx.norm(nu,'fro')
            prob = cvx.Problem(cvx.Minimize(cost),constraints)
            prob.solve()
            if prob.status != "optimal" :
                print(idx_s,prob.status)

            # Delta_list = Delta.value
            gamma_all = np.array([cost_list[i].value  for i in range(num_sample)])
            gamma[idx] = np.max(gamma_all)
            # print("gamma[{:}]".format(idx),gamma[idx],"nu",cvx.norm(nu).value)

        return gamma

class ODE_solution(object) :
    def __init__(self) :
        pass
    def setter(self,y,t) :
        self.y = y
        self.t = t

def RK4(odefun, tspan, y0,args,N_RK=10) :
    t = np.linspace(tspan[0],tspan[-1],N_RK)
    h = t[1] - t[0]
    iy = len(y0)
    y_sol = np.zeros((N_RK,iy))
    y_sol[0] = y0
    for idx in range(0,N_RK-1) :
        tk = t[idx]
        yk = y_sol[idx]
        k1 = odefun(tk,yk,*args)
        k2 = odefun(tk + h/2,yk + h/2*k1,*args)
        k3 = odefun(tk + h/2,yk + h/2*k2,*args)
        k4 = odefun(tk+h,yk+h*k3,*args)
        y_sol[idx+1] = yk + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    sol = ODE_solution()
    sol.setter(y_sol.T,t)
    return sol

def propagate_model(model,x0,u,w,delT) :
    def dfdt(t,x,u,w) :
        return np.squeeze(model.forward_uncertain(x,u,w))

    sol = solve_ivp(dfdt,(0,delT),x0,args=(u,w),method='RK45')
    # sol = RK4(dfdt,(0,delT),x0,args=(u,w),N_RK=5)
    xnext = sol.y[:,-1]
    return xnext

 
    
    
    
    
