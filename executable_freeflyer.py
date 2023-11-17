import matplotlib.pyplot as plt
import numpy as np
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
from model import FreeFlyer
from constraints import FreeFlyerConstraints
from cost import FreeFlyerCost
from matplotlib.patches import Ellipse
from utils.utils_alg import get_K_discrete
from utils.utils_plot import plot_ellipse_2D,plot_ellipse_3D,data_for_cylinder_along_z
from trajopt import trajopt
from funlopt import funlopt
from Lipschitz import Lipschitz
from Scaling import TrajectoryScaling
from jointsynthesis import jointsynthesis

# number of nodes, time horizon
N = 15
tf = 200
delT = tf/N

# obstacle setting
def get_H_obs(rx,ry) :
    return np.diag([1/rx,1/ry])
c_list = []
H_list = []
c1 = [0.5,1.2]
H1 = get_H_obs(0.8,0.8)
c_list.append(c1)
H_list.append(H1)
c2 = [2.5,2.0]
H2 = get_H_obs(0.8,0.8)
c_list.append(c2)
H_list.append(H2)
# for obstacle plotting
idx = 0
Xo1,Yo1,Zo1 = data_for_cylinder_along_z(c_list[idx][0],c_list[idx][1],1/H_list[idx][0,0],1/H_list[idx][1,1],3)
idx = 1
Xo2,Yo2,Zo2 = data_for_cylinder_along_z(c_list[idx][0],c_list[idx][1],1/H_list[idx][0,0],1/H_list[idx][1,1],3)

# freeflyer model
myModel = FreeFlyer.freeflyer('freeflyer','numeric_central')
ix = myModel.ix
iu = myModel.iu
iw = myModel.iw
iq = myModel.iq
ip = myModel.ip
C = myModel.C
D = myModel.D
E = myModel.E
G = myModel.G

# cost and constraint
myCost = FreeFlyerCost.freeflyer('Hello',ix,iu,N)
myConst = FreeFlyerConstraints.freeflyer('Hello',ix,iu)
myConst.set_obstacle(c_list,H_list)

# initial and final conditions
xi = np.array([0,0,3, 0,0,0, -np.deg2rad(30),np.deg2rad(25),np.deg2rad(5), 0,0,0])
xf = np.array([3,3,0, 0,0,0, 0,0,0, 0,0,0])
Qini = np.diag([0.2**2,0.2**2,0.2**2, \
                0.02**2,0.02**2,0.02**2, \
                np.deg2rad(5)**2,np.deg2rad(5)**2,np.deg2rad(5)**2, \
                np.deg2rad(0.1)**2,np.deg2rad(0.1)**2,np.deg2rad(0.1)**2])
Qf = Qini

# initial guess for trajectory optimization
x0 = np.zeros((N+1,ix))
for i in range(N+1) :
    x0[i] = (N-i)/N * xi + i/N * xf
# u0 = np.zeros((N+1,iu))
x0[:,3] = 1e-6
x0[:,4] = 1e-6
x0[:,5] = 1e-6
x0[:,9] = 1e-6
x0[:,10] = 1e-6
x0[:,11] = 1e-6
u0 = 1e-12*np.ones((N+1,iu))

# initial guess for funnel optimization
A,B,s,z,x_prop_n = myModel.diff_discrete_zoh(x0[0:N,:],u0[0:N,:],delT,tf) 
S = np.eye(ix)
R = np.eye(iu)
K0 = get_K_discrete(A,B,S,1e3*R,S,N,ix,iu)
Q0 = np.tile(Qini,(N+1,1,1))
Y0 = K0@Q0[:N]
betahat = np.ones(N+1)

# scaling
x_max = np.array([5,5,5,1,1,1,np.pi,np.pi,np.pi,np.deg2rad(1),np.deg2rad(1),np.deg2rad(1)])
x_min = np.zeros(ix)
u_max = np.array([0.02,0.02,0.02,100*1e-6,100*1e-6,100*1e-6]) 
u_min = np.zeros(iu)
traj_scaling = TrajectoryScaling(x_min,x_max,u_min,u_max,tf)
x_max = [0.1,0.1,0.1, 0.01,0.01,0.01, np.deg2rad(1),np.deg2rad(1),np.deg2rad(1), np.deg2rad(1),np.deg2rad(1),np.deg2rad(1)]
x_min = np.zeros(ix)
u_min = np.zeros(iu)
funl_scaling = TrajectoryScaling(x_min,x_max,u_min,u_max,tf)
funl_scaling.snu_p = 1e5

# trajectory optimization
max_iter = 20
traj_solver= trajopt('freeflyer',N,tf,max_iter,myModel,myCost,myConst,Scaling=traj_scaling,
              w_c=1e3,w_vc=1e2,w_tr=1e0,tol_vc=1e-6,tol_tr=1e-4,tol_dyn=1e-3,
              ignore_dpp=True,
              verbosity=False)
xfwd,_,xnom,unom,_,total_num_iter,_,_,_,_,history_nom = traj_solver.run(x0,u0,xi,xf)
tnom = np.array(range(N+1))*delT

# funnel optimization
alpha = 0.99
lambda_mu = 0.1
funl_w_tr = 0
w_Q = 1
w_K = 5*1e2
funl_solver = funlopt('freeflyer',ix,iu,iq,ip,iw,N,funl_scaling,
                      alpha=alpha,
                      lambda_mu=lambda_mu,
                      w_Q = w_Q,
                      w_K = w_K,
                      w_tr=funl_w_tr,
                      ignore_dpp=True,
                      solver="MOSEK",
                      flag_nonlinearity=True)
funl_solver.cvx_initialize(Qini,Qf)
lip_estimator = Lipschitz('freeflyer',ix,iu,iq,ip,iw,N,num_sample=256,
                            flag_uniform=True)
# get LQR to estimiate the Lipschitz constant
A,B,F,s,z,_ = myModel.diff_discrete_zoh_noise(xnom,unom,np.zeros((N,iw)),delT,tf) 
Ktmp = get_K_discrete(A,B,S,R,S,N,ix,iu)
Qtmp = np.tile(Qini,(N+1,1,1))
Ytmp = Ktmp@Qtmp[:N]
lip_estimator.initialize(xnom,unom,xfwd,Qtmp,Ktmp,A,B,C,D,E,F,G,myModel)
gammanew = lip_estimator.update_lipschitz_norm(myModel,delT)
Qnom,Knom,Ynom,status,funl_cost = funl_solver.solve(gammanew,Qtmp,Ytmp,A,B,C,D,E,F,G)

# parameter settings for joint synthesis
total_iter = 10
max_iter_trajopt = 1
tol_funl = 1e-8
tol_traj = 1e-8
tol_vc = 1e-8
tol_dyn = 1e-8
alpha = 0.99
lambda_mu = 0.1
funl_w_tr = 1e-8
w_Q = 1
w_K = 5*1e2

# trajopt and funnelopt classes
traj_solver= trajopt('freeflyer',N,tf,max_iter_trajopt,myModel,myCost,myConst,Scaling=traj_scaling,
              w_c=1,w_vc=1e2,w_tr=1e1,tol_vc=1e-6,tol_tr=1e-4,tol_dyn=1e-3,
              ignore_dpp=False,
              verbosity=False)
funl_solver = funlopt('freeflyer',ix,iu,iq,ip,iw,N,funl_scaling,
                      alpha=alpha,
                      lambda_mu=lambda_mu,
                      w_Q = w_Q,
                      w_K = w_K,
                      w_tr=funl_w_tr,
                      ignore_dpp=True,
                      solver="MOSEK",
                      flag_nonlinearity=True)
funl_solver.cvx_initialize(Qini,Qf,num_const_state=2,num_const_input=2)
lip_estimator = Lipschitz('freeflyer',ix,iu,iq,ip,iw,N,num_sample=256,
                            flag_uniform=True)

# joint synthesis class
JS = jointsynthesis(myModel,traj_solver,funl_solver,lip_estimator,total_iter,
                   tol_traj,tol_funl,tol_vc,tol_dyn,verbosity=True)


# run
history = JS.run(xi,xf,xnom,unom,Qnom,Ynom,Knom)

# load results
xbar = history[-1]['x']
ubar = history[-1]['u']
Qbar = history[-1]['Q']
Ybar = history[-1]['Y']
Kbar = history[-1]['K']

# plot
import matplotlib.font_manager as font_manager
csfont = {'fontsize':15,'fontname':'Times New Roman'}
font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=15)
fig = plt.figure(1,figsize=(12,5))
ax = fig.add_subplot(121)
for ce,H in zip(c_list,H_list) :
    rx = 1/H[0,0]
    ry = 1/H[1,1]
    circle1 = Ellipse((ce[0],ce[1]),rx*2,ry*2,color='tab:red',alpha=0.5,fill=True)
    ax.add_patch(circle1)
for i in range(N+1) :
    plot_ellipse_2D(ax,Qbar[i],xbar[i])
ax.set_xlabel('X (m)',**csfont)
ax.set_ylabel('Y (m)',**csfont)
ax.plot(xbar[:,0],xbar[:,1],color='tab:blue')
# plt.legend(fontsize=15)

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(Xo1, Yo1, Zo1, alpha=0.5,color='tab:red')
ax.plot_surface(Xo2, Yo2, Zo2, alpha=0.5,color='tab:red')
ax.set_xlabel('X (m)',**csfont)
ax.set_ylabel('Y (m)',**csfont)
ax.set_zlabel('Z (m)',**csfont)
ax.plot(xbar[:, 0],xbar[:, 1], xbar[:, 2],'o-',color='tab:blue')
for i in range(N+1) :
        plot_ellipse_3D(ax,Qbar[i],xbar[i])
ax.view_init(50, -60)
plt.show()