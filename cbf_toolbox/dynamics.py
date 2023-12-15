# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
# 
# This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.
# 
# © 2023 Massachusetts Institute of Technology.
# 
# Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
# 
# The software/firmware is provided to you on an As-Is basis
# 
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
#
# Author: Andrew Schoer, andrew.schoer@ll.mit.edu

import numpy as np
from math import sin, cos

class Dynamics:
    """
    Class to represent the affine dynamics
    x' = f(x) + g(x)u
    """
    def __init__(self,n,m,f,g) -> None:
        self.n = n # Dimension of state vector
        self.m = m # Dimension of input vector u
        self.f = f # Drift function
        self.g = g # Affine control function
        
    def dx(self, x, u):
        return self.f(x) + self.g(x) @ u

    def step(self, x, u, dt=0.1):
        return x + dt*self.dx(x,u)


class SingleIntegrator(Dynamics):
    '''
    Single integrator dynamics parameterized by the
    state vector length (input vector assumed to be identical)
    '''
    def __init__(self, n):
        '''
        Args:
            n: dimension of state vector x and input vector u
        '''
        f = lambda x : 0
        g = lambda x : np.eye(n)
        super().__init__(n, n, f, g)

    def __repr__(self):
        return f'SingleIntegrator{self.n}d'


def SingleIntegrator2d():
    return SingleIntegrator(2)

def SingleIntegrator3d():
    return SingleIntegrator(3)


class UnicycleNID(Dynamics): # NID - Near-Identity Diffeomorphism
    def __init__(self, l, theta) -> None:
        self.l = l # Offset to control point
        self.theta = theta # Initial pointing angle
        self.R = lambda theta: np.array([[cos(theta), -l*sin(theta)],[sin(theta), l*cos(theta)]])
        
        n = 3 # [x,y,theta]'
        m = 2 # [v,w]'
        f = lambda x : 0
        g = lambda x : np.array([[cos(x[2]), 0],[sin(x[2]), 0], [0, 1]])
        super().__init__(n, m, f, g)
        
    def dx(self, x, u):
        if len(x) == 2:
            x = np.append(x,self.theta)
        return self.R(x[2]).dot(u)
    
    def step(self, x, u, dt=0.1):
        theta0 = self.theta
        s0 = x[:2] + self.l * np.array([cos(theta0),sin(theta0)])
        s1 = s0 + dt*self.dx(x,u)
        self.theta = np.array(theta0 + dt*u[1])
        pos = s1 - self.l * np.array([cos(self.theta),sin(self.theta)])
        return pos
        # return np.hstack((pos,theta1))


if __name__ == '__main__':
    dyn = UnicycleNID(0.25, 0)
    x = np.array([0,0,0])
    u = np.array([0,np.pi/4])
    dt = 0.1
    
    for i in range(10):
        print(x)
        x = dyn.step(x,u,dt)
    print(x)