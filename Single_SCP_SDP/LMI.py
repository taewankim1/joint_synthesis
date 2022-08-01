import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np
import cvxpy as cvx
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

import cost
import model
import IPython

from Scaling import TrajectoryScaling

class Q_update :
    def __init__(self,ix,iu,iq,ip,iw,N,delT,S,R,w_tr=1,w_vc=1e3) :
        self.ix = ix
        self.iu = iu
        self.iq = iq
        self.ip = ip
        self.iw = iw
        self.N = N
        self.small = 1e-12
        self.delT = delT
        self.w_tr = w_tr
        self.w_vc = w_vc

        self.myScale = TrajectoryScaling()
    def initialize(self,x,u,e_prop,A,B,C,D,E,F,G) :
        ix,iu,N = self.ix,self.iu,self.N
        self.myScale.update_scaling_from_traj(x,u)
        self.Sx,self.iSx,self.sx,self.Su,self.iSu,self.su = self.myScale.get_scaling()

        self.Q_list = []
        self.Y_list = []
        for i in range(N+1) :
            self.Q_list.append(cvx.Variable((ix,ix), PSD=True))
            if i < N :
                self.Y_list.append(cvx.Variable((iu,ix)))
        self.nu_Q = cvx.Variable(N+1)
        self.nu_K = cvx.Variable(N)
        self.nu_p = cvx.Variable(1)
        self.vc = cvx.Variable(N)

        self.x = x
        self.u = u
        self.e_prop = e_prop

        self.A = A
        self.B = B 
        self.C = C 
        self.D = D 
        self.E = E 
        self.F = F 
        self.G = G

    def solve(self,alpha,gamma,Qini,Qf,Qbar,Ybar) :
        ix,iu,N,delT = self.ix,self.iu,self.N,self.delT
        iq,ip,iw = self.iq,self.ip,self.iw

        lambda_mu = 0.1
        lambda_zeta = 0.1
        constraints = []

        for i in range(N) :
            Qi = self.Sx@self.Q_list[i]@self.Sx
            Yi = self.Su@self.Y_list[i]@self.Sx
            Qi_next = self.Sx@self.Q_list[i+1]@self.Sx
            # Q_dot = (Qi_next - Qi) / delT
            
            LMI11 = alpha*Qi - lambda_mu * Qi - lambda_zeta * Qi
            LMI21 = np.zeros((ip,ix))
            LMI31 = np.zeros((iw,ix))
            LMI41 = np.zeros((ix,ix))
            LMI51 = self.A[i]@Qi+self.B[i]@Yi
            LMI61 = self.C@Qi+self.D@Yi
            
            LMI22 = self.nu_p * np.eye(ip)
            LMI32 = np.zeros((iw,ip))
            LMI42 = np.zeros((ix,ip))
            LMI52 = self.nu_p*self.E
            LMI62 = np.zeros((iq,ip))

            LMI33 = lambda_mu * np.eye(iw)
            LMI43 = np.zeros((ix,iw))
            LMI53 = self.F[i]
            LMI63 = self.G

            LMI44 = lambda_zeta * np.eye(ix)
            LMI54 = self.e_prop[i] * np.eye(ix)
            LMI64 = np.zeros((iq,ix))

            LMI55 = Qi_next
            LMI65 = np.zeros((iq,ix))

            LMI66 = self.nu_p * (1/gamma[i]**2) * np.eye(iq)


            row1 = cvx.hstack((LMI11,LMI21.T,LMI31.T,LMI41.T,LMI51.T,LMI61.T))
            row2 = cvx.hstack((LMI21,LMI22,LMI32.T,LMI42.T,LMI52.T,LMI62.T))
            row3 = cvx.hstack((LMI31,LMI32,LMI33,LMI43.T,LMI53.T,LMI63.T))
            row4 = cvx.hstack((LMI41,LMI42,LMI43,LMI44,LMI54.T,LMI64.T))
            row5 = cvx.hstack((LMI51,LMI52,LMI53,LMI54,LMI55,LMI65.T))
            row6 = cvx.hstack((LMI61,LMI62,LMI63,LMI64,LMI65,LMI66))
            LMI = cvx.vstack((row1,row2,row3,row4,row5,row6))
            

            # constraints.append(LMI + self.vc[i] * np.eye(ix+ip+iw+ix+ix+iq) >> 0)
            constraints.append(LMI  >> 0)
            constraints.append(self.nu_p>=self.small)
            
        for i in range(N+1) :
            Qi = self.Sx@self.Q_list[i]@self.Sx
            constraints.append(Qi << self.nu_Q[i]*np.eye(ix))
            constraints.append(Qi >> np.eye(ix)*self.small) # PD
            
        for i in range(N) :
            Yi = self.Su@self.Y_list[i]@self.Sx
            Qi = self.Sx@self.Q_list[i]@self.Sx
            tmp1 = cvx.hstack((self.nu_K[i]*np.eye(iu),Yi))
            tmp2 = cvx.hstack((Yi.T,Qi))
            constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)
            
        # initial condition
        Qi = self.Sx@self.Q_list[0]@self.Sx    
        constraints.append(Qi >> Qini)
        # final condition
        Qi = self.Sx@self.Q_list[-1]@self.Sx   
        constraints.append(Qi << Qf )

        # trust region
        objective_tr = []
        for i in range(N) :
            objective_tr.append(cvx.norm(self.Q_list[i]-self.iSx@Qbar[i]@self.iSx,'fro') 
                + cvx.norm(self.Y_list[i]-self.iSu@Ybar[i]@self.iSx,'fro'))
        objective_tr.append(cvx.norm(self.Q_list[-1]-self.iSx@Qbar[-1]@self.iSx,'fro'))
        
        l = cvx.sum(self.nu_Q) + cvx.sum(self.nu_K) # + self.w_tr * cvx.sum(l_t) # + self.w_vc*cvx.norm(self.vc)
        l_t = cvx.sum(objective_tr)

        l_all = l + self.w_tr*l_t

        prob = cvx.Problem(cvx.Minimize(l_all), constraints)
        prob.solve(solver=cvx.MOSEK)#verbose=False,solver=cvx.ECOS,warm_start=True)

        # l_t = []
        # for i in range(N) :
        #     l_t.append(cvx.norm(self.Q_list[i]-self.iSx@Qbar[i]@self.iSx,'fro') 
        #         + cvx.norm(self.Y_list[i]-self.iSu@Ybar[i]@self.iSx,'fro'))
        # l_t.append(cvx.norm(self.Q_list[-1]-self.iSx@Qbar[-1]@self.iSx,'fro'))
        
        # l = cvx.sum(self.nu_Q) + cvx.sum(self.nu_K) + self.w_tr * cvx.sum(l_t) + self.w_vc*cvx.norm(self.vc)
        # # l = 1
        # prob = cvx.Problem(cvx.Minimize(l), constraints)
        # prob.solve(solver=cvx.MOSEK)#verbose=False,solver=cvx.ECOS,warm_start=True)

        # cost_total = l.value
        # cost_QK = (cvx.sum(self.nu_Q) + cvx.sum(self.nu_K) ).value
        # cost_tr = cvx.sum(l_t).value
        # cost_vc = cvx.norm(self.vc).value
        # print("total_cost      cost         ||vc||     ||tr||")
        # print("{:10.3f}{:12.3g}{:12.3g}{:12.3g}".format(cost_total,cost_QK,cost_vc,cost_tr))

        Qnew = []
        Ynew = []
        for i in range(N+1) :
            Qnew.append(self.Sx@self.Q_list[i].value@self.Sx)
            if i < N :
                Ynew.append(self.Su@self.Y_list[i].value@self.Sx)
        Knew = []
        for i in range(N) :
            K = Ynew[i]@np.linalg.inv(Qnew[i])
            Knew.append(K)
        Knew = np.array(Knew)
        Qnew = np.array(Qnew)
        Ynew = np.array(Ynew)

        return Qnew,Knew,Ynew,prob.status,l.value

    def update_beta(self,Q,K,gamma,alpha) :
        ix,iu,N,delT = self.ix,self.iu,self.N,self.delT
        iq,ip,iw = self.iq,self.ip,self.iw
        
        beta_hat = np.zeros(N+1)
        beta_hat[0] = 1
        beta = np.zeros(N+1)
        for idx_beta in range(N) :
            A_cl = self.A[idx_beta] + self.B[idx_beta]@K[idx_beta]
            C_cl = self.C + self.D@K[idx_beta]

            S0 = np.hstack((A_cl,self.E,self.F[idx_beta])).T@np.linalg.inv(Q[idx_beta+1])@np.hstack((A_cl,self.E,self.F[idx_beta]))
            # print_np(S0)

            S1 = np.block([[np.linalg.inv(Q[idx_beta]),np.zeros((ix,ip+iw))],
                        [np.zeros((ip+iw,ix+ip+iw))]])
            # print_np(S1)
            S2_tmp = np.block([[C_cl,np.zeros((iq,ip)),self.G],
                            [np.zeros((ip,ix)),np.eye(ip),np.zeros((ip,iw))]])
            S2_tmp2 = np.block([[-(gamma[idx_beta]**2)*np.eye(iq),np.zeros((iq,ip))],[np.zeros((ip,iq)),np.eye(iq)]])
            S2 = S2_tmp.T@S2_tmp2@S2_tmp
            # print_np(S2)
            S3 = np.block([[np.zeros((ix+ip,ix+ip+iw))],
                        [np.zeros((iw,ix+ip)),np.eye(iw)]])
            # print_np(S3)

            lambda1 = cvx.Variable(1)
            lambda2 = cvx.Variable(1)
            lambda3 = cvx.Variable(1)

            constraints = []
            constraints.append(lambda1 >= 0)
            constraints.append(lambda2 >= 0)
            constraints.append(lambda3 >= 0)
            constraints.append(S0 - lambda1*S1 - lambda2*S2 - lambda3*S3 << 0)
            cost = lambda1 + lambda3

            prob = cvx.Problem(cvx.Minimize(cost), constraints)
            prob.solve(solver=cvx.MOSEK)
            beta_hat[idx_beta+1] = prob.value
            # print(idx_beta+1, prob.value)
        for idx_beta in range(N+1) :
            if idx_beta == 0 :
                beta[idx_beta] = 1
            elif idx_beta == 1 :
                beta[idx_beta] = beta_hat[idx_beta]
            else :
                beta[idx_beta] = max(alpha*beta[idx_beta-1],beta_hat[idx_beta])

        return beta






#     LMI11 = Qi@A[i].T+Yi.T@B[i].T+A[i]@Qi+B[i]@Yi-Q_dot+lambda_w*Qi
#     LMI12 = E
#     LMI13 = F
#     LMI14 = Qi@C.T+Yi.T@D.T
#     LMI15 = np.zeros((ix,iq*ip))
#     LMI16 = Qi@Cv.T+Yi.T@Dv.T
    
#     LMI22 = np.zeros((iq*ip,iq*ip))
#     LMI23 = np.zeros((iq*ip,iw))
#     LMI24 = np.zeros((iq*ip,iq*ip))
#     LMI25 = np.eye(iq*ip)
#     LMI26 = np.zeros((iq*ip,ix+iu))
    
#     LMI33 = -lambda_w * np.eye(iw)
#     LMI34 = G.T
#     LMI35 = np.zeros((iw,iq*ip))
#     LMI36 = np.zeros((iw,ix+iu))
    
#     LMI44 = -1/lambda_p * np.diag(1/gamma[i]**2)
#     LMI45 = np.zeros((iq*ip,iq*ip))
#     LMI46 = np.zeros((iq*ip,ix+iu))
    
#     LMI55 = -1/lambda_p * -np.eye(iq*ip)
#     LMI56 = np.zeros((iq*ip,ix+iu))
    
#     LMI66 = -np.eye(ix+iu)
    
#     row1 = cvx.hstack((LMI11,LMI12,LMI13,LMI14,LMI15,LMI16))
#     row2 = cvx.hstack((LMI12.T,LMI22,LMI23,LMI24,LMI25,LMI26))
#     row3 = cvx.hstack((LMI13.T,LMI23.T,LMI33,LMI34,LMI35,LMI36))
#     row4 = cvx.hstack((LMI14.T,LMI24.T,LMI34.T,LMI44,LMI45,LMI46))
#     row5 = cvx.hstack((LMI15.T,LMI25.T,LMI35.T,LMI45.T,LMI55,LMI56))
#     row6 = cvx.hstack((LMI16.T,LMI26.T,LMI36.T,LMI46.T,LMI56.T,LMI66))
#     LMI = cvx.vstack((row1,row2,row3,row4,row5,row6))

#     row1 = cvx.hstack((LMI11,LMI13))
#     row2 = cvx.hstack((LMI13.T,LMI33))
#     LMI = cvx.vstack((row1,row2))
    
#     row1 = cvx.hstack((LMI11,LMI13,LMI16))
#     row2 = cvx.hstack((LMI13.T,LMI33,LMI36))
#     row3 = cvx.hstack((LMI16.T,LMI36.T,LMI66))
#     LMI = cvx.vstack((row1,row2,row3))
################################################################################################







################################################################################################
#     LMI11 = Qi@A[i].T+Yi.T@B[i].T+A[i]@Qi+B[i]@Yi-Q_dot+(alpha+lambda_w)*Qi
#     LMI12 = E*nu_p
#     LMI13 = F
#     LMI14 = (Qi@C.T+Yi.T@D.T)@np.diag(gamma[i])
#     LMI22 = nu_p*-np.eye(iq*ip)
#     LMI23 = np.zeros((iq*ip,iw))
#     LMI24 = np.zeros((iq*ip,iq*ip))
#     LMI33 = -lambda_w * np.eye(iw)
#     LMI34 = G.T
#     LMI44 = -nu_p * np.eye(iq*ip)
    
#     row1 = cvx.hstack((LMI11,LMI12,LMI13,LMI14))
#     row2 = cvx.hstack((LMI12.T,LMI22,LMI23,LMI24))
#     row3 = cvx.hstack((LMI13.T,LMI23.T,LMI33,LMI34))
#     row4 = cvx.hstack((LMI14.T,LMI24.T,LMI34.T,LMI44))
#     LMI = cvx.vstack((row1,row2,row3,row4))

#     row1 = cvx.hstack((LMI11,LMI12,LMI14))
#     row2 = cvx.hstack((LMI12.T,LMI22,LMI24))
#     row3 = cvx.hstack((LMI14.T,LMI24.T,LMI44))
#     LMI = cvx.vstack((row1,row2,row3))
    
#     row1 = cvx.hstack((LMI11,LMI13))
#     row2 = cvx.hstack((LMI13.T,LMI33))
#     LMI = cvx.vstack((row1,row2))
################################################################################################