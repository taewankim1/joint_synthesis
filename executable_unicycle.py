import matplotlib.pyplot as plt
import numpy as np

from model import UnicycleModel
from cost import UnicycleCost
from constraints import UnicycleConstraints
from utils.utils_alg import get_K_discrete
from utils.utils_plot import plot_traj_set

from trajopt import trajopt
from funlopt import funlopt
from Lipschitz import Lipschitz
from Scaling import TrajectoryScaling
from jointsynthesis import jointsynthesis

N = 30
tf = 10
delT = tf/N

# obstacle setting
def get_H_obs(rx,ry) :
    return np.diag([1/rx,1/ry])
c_list = []
H_list = []
c_list.append([1.8,0.3])
H_list.append(get_H_obs(0.5,0.5))
c_list.append([2*2/3,3.5])
H_list.append(get_H_obs(0.5,0.5))
c_list.append([4.2,0.7])
H_list.append(get_H_obs(0.5,0.5))
c_list.append([3.8,3.5])
H_list.append(get_H_obs(0.5,0.5))
c_list.append([10*2/3,0.5])
H_list.append(get_H_obs(0.5,0.5))
c_list.append([6.5,4])
H_list.append(get_H_obs(0.5,0.5))
c_list.append([4*2/3,1.7])
H_list.append(get_H_obs(0.5,0.5))
c_list.append([8*2/3,2.3])
H_list.append(get_H_obs(0.5,0.5))
c_list.append([0.5,2])
H_list.append(get_H_obs(0.5,0.5))
c_list.append([8,2.3])
H_list.append(get_H_obs(0.5,0.5))

# type of unicycle model
type_model = 1
if type_model == 1 :
    myModel = UnicycleModel.unicycle1('unicycle','numeric_central')
elif type_model == 2 :
    myModel = UnicycleModel.unicycle2('unicycle','numeric_central')
elif type_model == 3 :
    myModel = UnicycleModel.unicycle3('unicycle','numeric_central')
ix = myModel.ix
iu = myModel.iu
iw = myModel.iw
iq = myModel.iq
ip = myModel.ip
C = myModel.C
D = myModel.D
E = myModel.E
G = myModel.G

# cost and constraints
myCost = UnicycleCost.unicycle('Hello',ix,iu,N,weight_factor_omega=1)
myConst = UnicycleConstraints.UnicycleConstraints('Hello',ix,iu,vmax=1.5,wmax=1.0,wmin=-1.0)
myConst.set_obstacle(c_list,H_list)

# Initial and final
xi = np.zeros(3)
xi[0] = 0.0
xi[1] = 0.0 
xi[2] = 0.0
xf = np.zeros(3)
xf[0] = 8.0
xf[1] = 4.0
xf[2] = np.deg2rad(30)
Qini = np.diag([0.2**2,0.2**2,np.deg2rad(10)**2])
Qf = np.diag([0.2**2,0.2**2,np.deg2rad(10)**2])

# initial guess - straight line
x0 = np.zeros((N+1,ix))
for i in range(N+1) :
    x0[i] = (N-i)/N * xi + i/N * xf
u0 = np.zeros((N+1,iu))

# initial guess for Q and K by LQR
A,B,s,z,x_prop_n = myModel.diff_discrete_zoh(x0[0:N,:],u0[0:N,:],delT,tf) 
S = np.eye(ix)
R = 1*np.eye(iu)
K0 = get_K_discrete(A,B,S,R,S,N,ix,iu)
Q0 = np.tile(np.diag([0.2**2,0.2**2,np.deg2rad(10)**2]),(N+1,1,1))
Y0 = K0@Q0[:N]
betahat = np.ones(N+1)

# scaling for decision variables
x_max = np.array([10,10,np.pi])
x_min = np.zeros(ix)
u_max = np.array([5,5]) 
u_min = np.zeros(iu)
traj_scaling = TrajectoryScaling(x_min,x_max,u_min,u_max,tf)
x_max = np.array([1,1,np.pi])
x_min = np.zeros(ix)
u_max = np.array([5,5]) 
u_min = np.zeros(iu)
funl_scaling = TrajectoryScaling(x_min,x_max,u_min,u_max,tf)
funl_scaling.snu_p = 1

# simulation setting
total_iter = 40
max_iter_trajopt = 1

# tolerance
tol_funl = 1e-8
tol_traj = 1e-8
tol_vc = 1e-8
tol_dyn = 1e-8

# parameters
alpha = 0.99
lambda_mu = 0.2
funl_w_tr = 1e-8

# trajopt and funelopt classes
traj_solver= trajopt('unicycle',N,tf,max_iter_trajopt,myModel,myCost,myConst,Scaling=traj_scaling,
              w_c=1,w_vc=1e3,w_tr=1e-1,tol_vc=1e-6,tol_tr=1e-4,tol_dyn=1e-3,verbosity=False)
funl_solver = funlopt('unicycle',ix,iu,iq,ip,iw,N,funl_scaling,
                      alpha=alpha,
                      lambda_mu=lambda_mu,
                      w_tr=funl_w_tr,
                      flag_nonlinearity=True)
funl_solver.cvx_initialize(Qini,Qf,num_const_state=len(c_list),num_const_input=4)
lip_estimator = Lipschitz('unicycle',ix,iu,iq,ip,iw,N,num_sample=100,flag_uniform=True)

JS = jointsynthesis(myModel,traj_solver,funl_solver,lip_estimator,total_iter,
                   tol_traj,tol_funl,tol_vc,tol_dyn,verbosity=True)


history = JS.run(xi,xf,x0,u0,Q0,Y0,K0)

xbar = history[-1]['x']
ubar = history[-1]['u']
Qbar = history[-1]['Q']
Ybar = history[-1]['Y']
Kbar = history[-1]['K']

fS = 20
plt.figure(0,figsize=(15,7))
plot_traj_set(xbar,ubar,c_list,H_list,Qbar,xi=xi,xf=xf,Qi=Qini,Qf=Qf,plt=plt,flag_label=True,fS=fS)
plt.legend(fontsize=fS)
plt.grid(True)
plt.rcParams["font.family"] = "Times New Roman"
plt.axis([-1.0, 9.0, -1.0, 5.0])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
