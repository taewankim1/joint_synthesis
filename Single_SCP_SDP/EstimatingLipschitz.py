import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np
import cvxpy as cvx
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))
from utils_alg import get_sample_eta_w,propagate_model

class Lipschitz :
    def __init__(self,ix,iu,iq,ip,iw,N) :
        self.ix = ix
        self.iu = iu
        self.iq = iq
        self.ip = ip
        self.iw = iw
        self.N = N

    def initialize(self,xbar,ubar,xprop,Qbar,Kbar,A,B,C,D,E,F,G,myModel,zs_sample,zw_sample)  :
        ix,iu,N = self.ix,self.iu,self.N
        ip,iq,iw = self.ip,self.iq,self.iw

        # self.Delta = cvx.Variable((ip*iq,ip*iq),diag=True)
        # self.nu = cvx.Variable(ix)

        # self.LHS = cvx.Parameter(ix)
        # # Ecvx = cvx.Parameter((ix,iq*ip))
        # self.mu = cvx.Parameter(iq*ip)

        # constraints = []
        # constraints.append(self.LHS == E@self.Delta@self.mu + self.nu)
        # cost = cvx.norm(self.Delta,'fro') + 1e4*cvx.norm(self.nu)
        # self.prob = cvx.Problem(cvx.Minimize(cost),constraints)
        # assert self.prob.is_dcp(dpp=True)

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
        assert len(zs_sample) == len(zw_sample)
        self.zs_sample = zs_sample
        self.zw_sample = zw_sample

    def update_lipschitz(self,myModel,delT) :
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

            constraints = []
            Delta_list = []
            gamma_val = []
            max_val = 0

            xnew_prop = xprop[idx]
            for idx_s in range(num_sample) :
                Delta = cvx.Variable((ip,iq))
                nu = cvx.Variable(ix)
                es,ws = eta_sample[idx_s],w_sample[idx_s] # samples on surface of ellipse
                # temporary
                ws *= 0

                xii = Kbar[idx]@es

                xs_prop = propagate_model(myModel,xbar[idx]+es,ubar[idx]+xii,delT)
                eta_prop = xs_prop - xbar[idx+1]

                LHS = eta_prop - (A[idx]+B[idx]@Kbar[idx])@es - F[idx]@ws + xbar[idx+1] - xnew_prop
                if LHS[2] > 1e-6 :
                    print("LHS[2] > 1e-6: is this possible?")

                mu = (C+D@Kbar[idx])@es + G@ws

                constraints.append(LHS == E@Delta@mu + 0*nu)
                cost = cvx.norm(Delta,'fro') + 1e4*cvx.norm(nu)
                prob = cvx.Problem(cvx.Minimize(cost),constraints)
                
                prob.solve()
                if prob.status != "optimal" :
                    print(idx_s,self.prob.status)
                if prob.value >= max_val :
                    idx_max = idx_s
                    max_val = prob.value 
                Delta_list.append(Delta.value)
                gamma_val.append(cvx.norm(Delta,2).value)
            Delta_list = np.array(Delta_list)
            gamma_val = np.array(gamma_val)

            gamma[idx] = np.max(gamma_val)
            print("gamma[{:}]".format(idx),gamma[idx],"nu",cvx.norm(nu).value)

        return gamma
    
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
            num_sample = 100
            eta_sample,w_sample = get_sample_eta_w(Qbar[idx],self.zs_sample,self.zw_sample) 
            num_sample = len(self.zs_sample)

            Delta = cvx.Variable((num_sample,ip*iq))
            nu = cvx.Variable((num_sample,ix))

            constraints = []
            cost_list = []
            deltanorm_list = []

            xnew_prop = xprop[idx]
            for idx_s in range(num_sample) :
                es,ws = eta_sample[idx_s],w_sample[idx_s] # samples on surface of ellipse
                # temporary
                ws *= 0

                xii = Kbar[idx]@es

                xs_prop = propagate_model(myModel,xbar[idx]+es,ubar[idx]+xii,delT)
                eta_prop = xs_prop - xbar[idx+1]

                LHS = eta_prop - (A[idx]+B[idx]@Kbar[idx])@es - F[idx]@ws + xbar[idx+1] - xnew_prop
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
            xnew_prop = xprop[idx]
            for idx_s in range(num_sample) :
                es,ws = eta_sample[idx_s],w_sample[idx_s] # samples on surface of ellipse
                # temporary
                ws *= 0

                xii = Kbar[idx]@es

                xs_prop = propagate_model(myModel,xbar[idx]+es,ubar[idx]+xii,delT)
                eta_prop = xs_prop - xbar[idx+1]

                LHS = eta_prop - (A[idx]+B[idx]@Kbar[idx])@es - F[idx]@ws + xbar[idx+1] - xnew_prop
                if LHS[2] > 1e-6 :
                    print("LHS[2] > 1e-6: is this possible?")

                mu = (C+D@Kbar[idx])@es + G@ws

                gamma_val.append(np.linalg.norm(LHS)/np.linalg.norm(mu))
            gamma_val = np.array(gamma_val)

            gamma[idx] = np.max(gamma_val)
            # print("gamma[{:}]".format(idx),gamma[idx])

        return gamma
    
    
    
    
