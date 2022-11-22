import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
#     print ("Values are: \n%s" % (x))

import os
import sys

from model import UnicycleModel
from cost import UnicycleCost
from constraints import UnicycleConstraints
from scipy.integrate import solve_ivp
from matplotlib.patches import Ellipse
from utils.utils_alg import get_neighbor_vec,get_K_discrete
from utils.utils_plot import plot_traj,plot_traj_set,plot_state_input
from Single_PTR import Single_PTR
from PTR import PTR
from utils.utils_alg import get_sample_eta_w,propagate_model
import cvxpy as cvx
from scipy.integrate import solve_ivp
from Lipschitz import Lipschitz
from funlopt import Q_update
from utils.utils_alg import forward_full_with_K,get_sample_trajectory
import scipy

ix = 3
iu = 2
iw = 2
iq = 2
ip = 2
N = 30
tf = 3
delT = tf/N

# time-invariant matrices
C = np.array([[0,0,1],[0,0,0]])
D = np.array([[0,0],[1,0]])
E = np.array([[1,0],[0,1],[0,0]])
G = np.zeros((iq,iw))

def get_H_obs(rx,ry) :
    return np.diag([1/rx,1/ry])
# obstacle
c_list = []
H_list = []
c1 = [1,2]
H1 = get_H_obs(0.75,1.5)
c_list.append(c1)
H_list.append(H1)
c2 = [4,3]
H2 = get_H_obs(0.75,1.5)
c_list.append(c2)
H_list.append(H2)

xi = np.zeros(3)
xi[0] = 0.0
xi[1] = 0.0 
xi[2] = 0

xf = np.zeros(3)
xf[0] = 5.0
xf[1] = 5.0
xf[2] = 0

Qini = np.diag([0.4**2,0.4**2,np.deg2rad(20)**2])
Qf = np.diag([0.5**2,0.5**2,np.deg2rad(20)**2])*1.5

myModel = UnicycleModel.unicycle('Hello',ix,iu,iw,'numeric_central')
myCost = UnicycleCost.unicycle('Hello',ix,iu,N)
myConst = UnicycleConstraints.UnicycleConstraints('Hello',ix,iu)
myConst.set_obstacle(c_list,H_list)

x0 = np.zeros((N+1,ix))
for i in range(N+1) :
    x0[i] = (N-i)/N * xi + i/N * xf
u0 = np.zeros((N+1,iu))

A,B,s,z,x_prop_n = myModel.diff_discrete_zoh(x0[0:N,:],u0[0:N,:],delT,tf) 
S = np.eye(ix)
R = np.eye(iu)
K0 = get_K_discrete(A,B,S,R,S,N,ix,iu)
Q0 = np.tile(np.diag([0.35**2,0.35**2,np.deg2rad(10)**2]),(N+1,1,1))
Y0 = K0@Q0[:N]
betahat = np.ones(N+1)

num_sample = 100
zs_sample = [] # sample in unit sphere will be projected to ellipse (Q_k)
zw_sample = [] 
    
# uniformly fixed
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

history = []
total_iter = 40
tol_funnel = 1e-3
tol_traj = 1e-4

print("STEP 0 : Start")
for idx_iter in range(total_iter) :
    print("====================================================")
    if idx_iter == 0 :
        xhat,uhat,Qhat,Yhat,Khat = x0,u0,Q0,Y0,K0
    
    print("STEP 1 : Nominal trajectory update")
    traj_solver= Single_PTR('unicycle',N,tf,1,myModel,myCost,myConst,
              w_c=1,w_vc=1e3,w_tr=1e1,tol_vc=1e-6,tol_tr=1e-4,verbosity=False)
    _,_,xnew,unew,total_num_iter,flag_boundary,traj_cost,traj_vc,traj_tr = traj_solver.run(xhat,uhat,xi,xf,Qhat,Khat)

    # discretization
    A,B,F,s,z,x_prop_n = myModel.diff_discrete_zoh_noise(xnew,unew,np.zeros((N,iw)),delT,tf) 
    sz = tf*s + z
    # propagation
    xprop,_ = traj_solver.forward_multiple(xnew,unew)
    e_prop = np.linalg.norm(xprop - xnew[1:],axis=1)
    
    print("STEP 2 : Lipschitz constant estimation")
    myM = Lipschitz(ix,iu,iq,ip,iw,N)
    myM.initialize(xnew,unew,xprop,Qhat,Khat,A,B,C,D,E,F,G,myModel,zs_sample,zw_sample)
    gamma = myM.update_lipschitz_norm(myModel,delT)
#     gamma = myM.update_lipschitz_parallel(myModel,delT)
#     print("mean of gamma",np.mean(gamma,0),"max of gamma",np.max(gamma,0),"var of gamma",np.var(gamma,0))

    print("STEP 3 : Funnel update via SDP")
    funl_solver = Q_update(ix,iu,iq,ip,iw,N,delT,myCost.S,myCost.R,w_tr=1e-1)
    funl_solver.initialize(xnew,unew,e_prop*0,A,B,C,D,E,F,G)
    alpha = 0.99
    Qnew,Knew,Ynew,status,funl_cost = funl_solver.solve(alpha,gamma,Qini,Qf,Qhat,Yhat)
#     print("LMI status:" + status)
    
    print("STEP 4 : update beta for invariance")
    betanew = funl_solver.update_beta(Qnew,Knew,gamma,alpha)
    for i in range(N+1) :
        Qnew[i] = Qnew[i] * betanew[i]
#     print("beta max {:}".format(np.max(betanew)))
    
    # measure the difference
    xdiff = np.sum(np.linalg.norm(xhat-xnew,axis=1)**2)
    udiff = np.sum(np.linalg.norm(uhat-unew,axis=1)**2)
    Qdiff = np.sum(np.array([np.linalg.norm(Qhat[i]-Qnew[i],ord='fro') for i in range(N+1)])**2)
    Kdiff = np.sum(np.array([np.linalg.norm(Khat[i]-Knew[i],ord='fro') for i in range(N)])**2)
    Ydiff = np.sum(np.array([np.linalg.norm(Yhat[i]-Ynew[i],ord='fro') for i in range(N)])**2)
    betadiff = np.linalg.norm(betanew-betahat)
    
    # update trajectory
    xhat = xnew
    uhat = unew
    Qhat = Qnew
    Khat = Knew
    Yhat = Ynew
    betahat = betanew
    
    # save trajectory
    traj = {}
    traj['x'] = xhat
    traj['u'] = uhat
    traj['Q'] = Qhat
    traj['Y'] = Yhat
    traj['K'] = Khat
    traj['gamma'] = gamma
    traj['traj_diff'] = xdiff + udiff
    traj['funl_diff'] = Qdiff + Kdiff
    traj['beta_diff'] = betadiff
    history.append(traj)
    
    print("iter| traj_cost | funl_cost |   vc   |   Delta_T   |   Delta_F   |e_prop|mean_gamma|max_beta")
    print("%4d %11.3f %11.3f %8.3g %13.3g %13.3g %6.3f %10.3f %8.3g" % ( idx_iter+1,
                        traj_cost,funl_cost,traj_vc,
                        traj['traj_diff'],traj['funl_diff'],
                        np.max(e_prop),
                        np.mean(gamma,0),
                        np.max(betanew),
                        ))

    
    
    if traj['traj_diff'] < tol_traj and traj['funl_diff'] < tol_funnel :
        print("SUCCESS")
        break
    else :
        print("Accept the step")

traj_diff_list = [history[i]['traj_diff'] for i in range(idx_iter+1)]
funl_diff_list = [history[i]['funl_diff'] for i in range(idx_iter+1)] 


xbar = history[-1]['x']
ubar = history[-1]['u']
Qbar = history[-1]['Q']
Ybar = history[-1]['Y']
Kbar = history[-1]['K']

traj_solver= PTR('unicycle',N,tf,30,myModel,myCost,myConst,
              w_c=1,w_vc=1e3,w_tr=1e1,tol_vc=1e-6,tol_tr=1e-4,verbosity=True)
_,_,xnom,unom,total_num_iter,flag_boundary,_,_,_ = traj_solver.run(x0,u0,xi,xf)

x0_sample = []
num_sample = 100
idx = 0
for i in range(num_sample) :
    z = np.random.randn(ix)
    z = z / np.linalg.norm(z)
    x_s = xbar[idx] + scipy.linalg.sqrtm(Qbar[idx])@z
    x0_sample.append(x_s)
    
tsam,xsam,usam,wsam = get_sample_trajectory(xi,x0_sample,xbar,ubar,Qbar,Ybar,myModel,N,ix,iu,iw,delT,flag_noise=True,discrete=True)

i_index = np.array([i+1 for i in range(idx_iter+1)])
fS = 20
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(i_index,traj_diff_list,'o-')
plt.plot(i_index,i_index*0+tol_traj,'--')
plt.xlabel('iteration number',fontsize=fS,fontname='Times New Roman')
plt.ylabel(r'$\Delta_T$',fontsize=fS,fontname='Times New Roman')
plt.yscale('log')
# plt.ylim([10**(-6), 10**(4)])
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.legend(fontsize=fS)
plt.subplot(122)
plt.plot(i_index,funl_diff_list,'o-')
plt.plot(i_index,i_index*0+tol_funnel,'--',label="tolerance")
plt.xlabel('iteration number',fontsize=fS,fontname='Times New Roman')
plt.ylabel(r'$\Delta_F$',fontsize=fS,fontname='Times New Roman')
plt.yscale('log')
# plt.ylim([10**(-6), 10**(4)])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.rcParams["font.family"] = "Times New Roman"
plt.legend(fontsize=fS)
plt.grid(True)
plt.show()

fS = 15
plt.figure(0,figsize=(7,7))
for xsam_e in xsam :
    plt.plot(xsam_e[:,0], xsam_e[:,1],'-',markersize=4.0, linewidth=1.0,alpha=0.4,color='tab:purple')
plot_traj_set(xbar,ubar,c_list,H_list,Qbar,xi=xi,xf=xf,Qi=Qini,Qf=Qf,plt=plt,flag_label=True)
plt.plot(1e3,1e3,'-',color='tab:purple',label='samples')
plt.plot(xnom[:,0],xnom[:,1],'-.',color='tab:brown',label='traj w/o funnel')
plt.legend(fontsize=fS)
plt.grid(True)
plt.rcParams["font.family"] = "Times New Roman"
plt.show()

fS = 20
plt.figure(0,figsize=(10,5))
alpha = 0.5
t_index = np.array(range(N+1))*delT
for i in range(num_sample) :
    tsam_e = tsam[i]
    xsam_e = xsam[i]
    usam_e = usam[i]
#     plot_state_input(tsam_e,xsam_e,usam_e,None,None,N,delT,alpha,plt,flag_step=False)

    plt.subplot(121)
    plt.plot(tsam_e, usam_e[:,0],color='tab:purple',alpha=alpha,linewidth=1.0)
    plt.subplot(122)
    plt.plot(tsam_e, usam_e[:,1],color='tab:purple',alpha=alpha,linewidth=1.0)
plt.subplot(121)
plt.plot(tsam_e, usam_e[:,0]*0+myConst.vmax,'-.',color='tab:red',alpha=1.0,linewidth=2.0,label='limit')
plt.step(t_index, [*ubar[:N,0],ubar[N-1,0]],'--',color='tab:orange',alpha=1.0,where='post',linewidth=2.0,label='nominal')
plt.plot(1e3, 1e3,'-',color='tab:purple',alpha=1.0,linewidth=1.0,label='samples')
plt.xlabel('time (s)', fontsize = fS)
plt.ylabel('$u_v$ (m/s)', fontsize = fS)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.axis([0.0, tf, -0.5, 4.5])
plt.grid(True)
plt.subplot(122)
plt.plot(tsam_e, usam_e[:,1]*0+myConst.wmax,'-.',color='tab:red',alpha=1.0,linewidth=2.0)
plt.plot(tsam_e, usam_e[:,1]*0+myConst.wmin,'-.',color='tab:red',alpha=1.0,linewidth=2.0,label='limit')
plt.step(t_index, [*ubar[:N,1],ubar[N-1,1]],'--',color='tab:orange',alpha=1.0,where='post',linewidth=2.0,label='nominal')
plt.plot(1e3, 1e3,'-',color='tab:purple',alpha=1.0,linewidth=1.0,label='samples')
plt.xlabel('time (s)', fontsize = fS)
plt.ylabel('$u_{\Theta}$ (rad/s)', fontsize = fS)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.axis([0.0, tf, -3, 3])
plt.legend(fontsize=fS)
plt.rcParams["font.family"] = "Times New Roman"
plt.grid(True)
plt.show()