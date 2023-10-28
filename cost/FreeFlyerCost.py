import matplotlib.pyplot as plt
import numpy as np
import time
import random
import cvxpy as cvx

def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

from cost.cost import OptimalcontrolCost

class freeflyer(OptimalcontrolCost):
    def __init__(self,name,ix,iu,N):
        super().__init__(name,ix,iu,N)
        self.ix = ix
        self.iu = iu
        self.N = N

    # def bc_final(self,xcvx,xf):
    #     h = []
    #     h.append(xcvx == xf)
    #     return h

    def estimate_cost_cvx(self,x,u,idx=None):
        # dimension
        cost_total = cvx.quad_form(u,np.eye(self.iu))
        
        return cost_total