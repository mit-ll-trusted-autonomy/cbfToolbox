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

from abc import ABC
import numpy as np
from cbf_toolbox.dynamics import *
from cbf_toolbox.geometry import Shape, Sphere, Point
from gurobipy import GRB
import matplotlib.pyplot as plt
from math import sin, cos

class Vertex(ABC):
    """Objects that are related to each other through an Edge

    The objects that are in the simulation (Agent, Obstacle, Goal) are types of Vertex and
    relate to each other through either a CLF or a CBF
    """
    def __init__(self, state, shape, dynamics, color='red'):
        self.state = state
        self.shape = shape
        self.dynamics = dynamics
        self.color = color
        
    def step(self,u=None,dt=0.1):
        """Move forward one time step"""
        if u is None:
            u = np.zeros(self.dynamics.m)
        self.state = self.dynamics.step(self.state,u,dt)

    def plot(self,ax):
        """Plot the Vertex object"""
        self.shape.plot(ax,self.state,color=self.color)

class Agent(Vertex):
    """Control inputs for an agent are solved for by the Gurobi model"""

    def __init__(self, state, shape, dynamics, safety=True, plot_arrows=False, plot_path=False, color='green', k=1.0, p=1.0):
        super().__init__(state, shape, dynamics, color)
        
        # Boolean params
        self.safety = safety
        self.plot_arrows = plot_arrows
        self.plot_path = plot_path

        # History
        self.trajectory = state
        self.u_hist = []
        
        # CBF params
        self.k = k
        self.p = p

    def add_grb_control_vars(self, m, upper_bounds=None, lower_bounds=None):
        u_dim = self.dynamics.m

        # Format the upper bound of the variables
        if upper_bounds is None:
            ubs = [GRB.INFINITY] * u_dim
        elif isinstance(upper_bounds, list):
            ubs = upper_bounds
        else:
            ubs = [upper_bounds] * u_dim

        # Format the lower bound of the variables
        if lower_bounds is None:
            lbs = [-GRB.INFINITY] * u_dim
        elif isinstance(lower_bounds, list):
            lbs = lower_bounds
        else:
            lbs = [lower_bounds] * u_dim

        # Add the variable to the model
        u = [m.addVar(lb=lbs[i], ub=ubs[i], vtype=GRB.CONTINUOUS, name='u{}'.format(i)) for i in range(u_dim)]
        self.u = np.array(u)
        m.update()

    @property
    def u(self):
        try:
            u = [ui.getAttr("x") for ui in self._u]
            return np.array(u)
        except:
            return self._u
        
    @u.setter
    def u(self,value):
        self._u = value

    @property
    def goal(self):
        """Get the goal"""
        return self._goal

    @goal.setter
    def goal(self,value):
        """Set the goal value. Make sure it is of type Vertex"""
        if not isinstance(value,Vertex) and value is not None:
            raise ValueError('Must pass in a Vertex object')
        value.color = self.color
        self._goal = value

    def add_vel_constr(self,m):
        """Adds velocity constraint through the dynamics object"""
        self.dynamics.add_vel_constr(m)

    def step(self, u=None, dt=0.1):
        """Steps the agent forward one time step"""
        if u is None:
            u = self.u
        super().step(u,dt)
        # Save control action
        self.trajectory = np.vstack([self.trajectory,self.state])
        self.u_hist.append(self.u)
        self.dt = dt

    def plot(self,ax):
        """Plot the agent"""
        super().plot(ax)

        if type(self.dynamics) is UnicycleNID:
            r = self.shape.radius
            theta = self.dynamics.theta
            x = self.state[0]
            y = self.state[1]
            dx = cos(theta) * r
            dy = sin(theta) * r
            plt.gca().arrow(x,y,dx,dy,length_includes_head=True, width=2*r*.05, head_width=2*r*.3, fc=self.color,ec=self.color)
        
        if self.plot_path:
            plt.plot(self.trajectory[:,0],self.trajectory[:,1],color=self.color, linewidth=5)

        if self.plot_arrows:
            self.plot_control()

        if self.goal is not None:
            self.goal.plot(ax)

    def plot_control(self):
        """Plot the arrows that represent safe control action and desired control/direction to goal"""
        state = self.state
        rad = self.shape.radius
        
        if self.u_ref is not None:
            u_ref = self.u_ref
            b_ref = state + (u_ref/np.linalg.norm(u_ref))*rad
            plt.arrow(b_ref[0],b_ref[1],u_ref[0],u_ref[1],head_width=0.2,width=0.05,ec='red',color='red',length_includes_head=True)
        else:
            dir = self.goal.state - state
            u_ref = dir / np.linalg.norm(dir) * np.linalg.norm(self.u)
            b_ref = state + (u_ref/np.linalg.norm(u_ref))*rad
            plt.arrow(b_ref[0],b_ref[1],u_ref[0],u_ref[1],head_width=0.2,width=0.05,ec='red',color='red',length_includes_head=True)

        u = self.u
        b = state + (u/np.linalg.norm(u))*rad
        plt.arrow(b[0],b[1],u[0],u[1],head_width=0.2,width=0.05,ec='green',color='green',length_includes_head=True)
        

class Obstacle(Vertex):
    """Object that relates to an agent through a CBF. Agent will avoid collision with obstacles"""
    def __init__(self, state, shape, dynamics, k=1.0, p=1.0):
        super().__init__(state, shape, dynamics)
        self.k = k
        self.p = p
        self.u = np.zeros(dynamics.m)

class Goal(Vertex):
    """Object that relates to an Agent through a CLF. Agent will move toward a goal"""
    def __init__(self, state, shape=Point(), dynamics=SingleIntegrator2d(), p=1.0, gamma=0.25, H=None):
        state = np.array(state, dtype=np.float32)
        self.p = p
        self.gamma = gamma
        self.H = H
        self.u = np.zeros(dynamics.m)
        super().__init__(state, shape, dynamics)
