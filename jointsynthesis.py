import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))
from utils.utils_plot import plot_traj_set

class jointsynthesis:
    def __init__(self,model,traj_solver,funl_solver,lip_estimator,total_iter,
        tol_traj=1e-8,
        tol_funl=1e-8,
        tol_vc=1e-8,
        tol_dyn=1e-8,
        verbosity = True
        ) :

        self.model = model
        self.traj_solver = traj_solver
        self.funl_solver = funl_solver
        self.lip_estimator = lip_estimator

        self.total_iter = total_iter

        self.tol_traj = tol_traj
        self.tol_funl = tol_funl
        self.tol_vc = tol_vc
        self.tol_dyn = tol_dyn

        self.verbosity = verbosity

    def run(self,xi,xf,x0,u0,Q0,Y0,K0) :
        N,delT,tf = self.traj_solver.N,self.traj_solver.delT,self.traj_solver.tf
        
        C,D,E,G = self.model.C,self.model.D,self.model.E,self.model.G
        ix,iu,iw,iq,ip = self.model.ix,self.model.iu,self.model.iw,self.model.iq,self.model.ip

        history = []
        for idx_iter in range(self.total_iter) :
            # save trajectory
            sub_history = {}
            if idx_iter == 0 :
                xhat,uhat,Qhat,Yhat,Khat,gammahat = x0,u0,Q0,Y0,K0,np.zeros(N)
            
            # STEP 1 : Nominal trajectory update
            xfwd,_,xnew,unew,_,total_num_iter,_,traj_cost,traj_vc,traj_tr,traj_history = self.traj_solver.run(xhat,uhat,xi,xf,Qhat,Khat)
            dyn_error = np.linalg.norm(xfwd - xnew,axis=1)
            sub_history['t_trajopt'] = traj_history[-1]['cvxopt']
            
            # discretization
            tic = time.time()
            A,B,F,s,z,_ = self.model.diff_discrete_zoh_noise(xnew,unew,np.zeros((N,iw)),delT,tf) 
            sub_history['t_derivs'] = time.time() - tic
            
            # STEP 2 : Lipschitz constant estimation
            tic = time.time()
            self.lip_estimator.initialize(xnew,unew,xfwd,Qhat,Khat,A,B,C,D,E,F,G,self.model)
            gammanew = self.lip_estimator.update_lipschitz_norm(self.model,delT)
            sub_history['t_Lipschitz'] = time.time() - tic
        #     print("mean of gamma",np.mean(gammanew,0),"max of gamma",np.max(gammanew,0),"var of gamma",np.var(gammanew,0))

            # STEP 3 : Funnel update via SDP
            tic = time.time()
            Qnew,Knew,Ynew,status,funl_cost = self.funl_solver.solve(gammanew,Qhat,Yhat,A,B,C,D,E,F,G)
            sub_history['t_funlopt'] = time.time() - tic
            
            # measure the difference
            xdiff = np.sum(np.linalg.norm(xhat-xnew,axis=1)**2)
            udiff = np.sum(np.linalg.norm(uhat-unew,axis=1)**2)
            Qdiff = np.sum(np.array([np.linalg.norm(Qhat[i]-Qnew[i],ord='fro') for i in range(N+1)])**2)
            Kdiff = np.sum(np.array([np.linalg.norm(Khat[i]-Knew[i],ord='fro') for i in range(N)])**2)
            Ydiff = np.sum(np.array([np.linalg.norm(Yhat[i]-Ynew[i],ord='fro') for i in range(N)])**2)
            gammadiff = np.sum(np.abs(gammahat-gammanew))
            
            # update trajectory
            xhat = xnew
            uhat = unew
            Qhat = Qnew
            Khat = Knew
            Yhat = Ynew
            gammahat = gammanew
        
            sub_history['x'] = xhat
            sub_history['u'] = uhat
            sub_history['Q'] = Qhat
            sub_history['Y'] = Yhat
            sub_history['K'] = Khat
            sub_history['gamma'] = gammahat
            sub_history['traj_diff'] = xdiff + udiff
            sub_history['funl_diff'] = Qdiff + Kdiff
            sub_history['gamma_diff'] = gammadiff
            sub_history['dyn_error'] = np.sum(dyn_error)
            sub_history['vc'] = traj_vc
            history.append(sub_history)
            
            flag_vc = traj_vc < self.tol_vc
            flag_traj_tr = sub_history['traj_diff'] < self.tol_traj
            flag_funl_tr = sub_history['funl_diff'] < self.tol_funl
            flag_dyn =  sub_history['dyn_error'] < self.tol_dyn
            
            if idx_iter == 0 and self.verbosity == True :
                print("|iter| traj_cost | funl_cost |   vc   |   Delta_T   |   Delta_F   | e_prop  |gamma diff|")
                print("|    |           |           | log10  |   log10     |   log10     | log10   |          |")
            if self.verbosity == True :
                print("|%-4d|%-11.3f|%-11.3f|%-5.3g(%-1d)|%-10.3g(%-1d)|%-10.3g(%-1d)|%-3.3f(%-1d)|%-10.8f|" % ( idx_iter+1,
                                traj_cost,funl_cost,
                                np.log10(traj_vc),flag_vc,
                                np.log10(sub_history['traj_diff']),flag_traj_tr,
                                np.log10(sub_history['funl_diff']),flag_funl_tr,
                                np.log10(np.sum(dyn_error)),flag_dyn,
                                sub_history['gamma_diff'],
                                ))
            if flag_vc and flag_traj_tr and flag_funl_tr and flag_dyn :
                print("SUCCESS")
                break
        return history
    


    