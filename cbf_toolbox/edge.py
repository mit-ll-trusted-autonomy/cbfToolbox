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

from abc import ABC, abstractmethod
from gurobipy import GRB
from jax import grad
import numpy as np
import matplotlib.pyplot as plt

class Edge(ABC):
    """Object to define the relationship between two Vertex objects"""

    def __init__(self,agent,vertex):
        self.agent = agent
        self.vertex = vertex

    @abstractmethod
    def plot(self):
        pass

class ReferenceControl(Edge):
    """Reference control to move in any direction"""
    def __init__(self, agent, u_ref):
        super().__init__(agent, None)

        self.u_ref = u_ref

    def add_constraints(self, m):
        u_vars = self.agent.u
        for idx,u_var in enumerate(u_vars):
            m.addConstr(u_var == self.u_ref[idx], 'u{}'.format(idx))
        m.update()

    def plot(self):
        pass

class CLF(Edge):
    """Control Lyapunov Function to govern the attraction between an Agent and Vertex"""
    def __init__(self, agent, goal):
        super().__init__(agent, goal)

        self.lyap = goal.shape.func
        self.v_hist = []
        self.gamma = goal.gamma
        self.p = goal.p
        self.H = goal.H

    def add_clf(self,m):
        """Adds CLF constraint to the Gurobi model"""
        if not hasattr(self,'delta'):
            # Add relaxation variable delta
            self.delta = m.addVar(vtype=GRB.CONTINUOUS, name='delta')
        delta = self.delta

        x_agent = self.agent.state
        u_agent = self.agent.u
        x_goal = self.vertex.state
        u_goal = self.vertex.u
        x = x_agent - x_goal
        u = u_agent - u_goal
        xdot = self.agent.dynamics.dx(x_agent,u_agent) - self.vertex.dynamics.dx(x_goal,u_goal)
        
        p = self.p
        gamma = self.gamma

        if self.H is None:
            self.H = np.identity(len(u_agent)) * 1
        H=self.H

        v = self.lyap(x)
        grad_v = np.array(grad(self.lyap, argnums=0)(x))
        # lf_v = grad_v.T.dot(H).dot(xdot)
        lf_v = grad_v.T.dot(xdot)
        
        constraint = (lf_v + gamma*v <= delta)
        cost_func = 0.5 * u.T.dot(H).dot(u) + gamma*grad_v.T.dot(H).dot(xdot) + p*delta*delta

        m.addConstr(constraint, 'clf')
        m.setObjective(cost_func + m.getObjective(), GRB.MINIMIZE)
        m.update()

        # Save data to plot
        self.v_hist.append(v)

    def plot(self, ax=None, color=None):
        """Plot the CLF value over time"""
        if ax is None:
            plt.cla()
            if color is None:
                plt.plot(self.v_hist,'-')
            else:
                plt.plot(self.v_hist,'-', color=color)
            plt.title('Lyapunov Function Value')
            plt.pause(.001) # Need to pause for plot to update
            plt.show()
        else:
            if color is None:
                ax.plot(self.v_hist,'-')
            else:
                ax.plot(self.v_hist,'-', color=color, linewidth=5)
        


class CBF(Edge):
    """Control Barrier Function to ensure an Agent does not collide with Vertex"""
    def __init__(self, agent, obstacle):
        super().__init__(agent, obstacle)
        self.barrier = obstacle.shape.func
        self.h_hist = []
        self.k = obstacle.k
        self.p = obstacle.p

    def add_cbf(self,m):
        """Adds the CBF constraint to the Gurobi model"""
        x_agent = self.agent.state
        u_agent = self.agent.u
        x_obs = self.vertex.state
        u_obs = self.vertex.u
        x = x_agent - x_obs
        xdot = self.agent.dynamics.dx(x_agent,u_agent) - self.vertex.dynamics.dx(x_obs,u_obs)
        agent_rad = self.agent.shape.radius

        k = self.k
        p = self.p

        h = self.barrier(x,agent_rad)
        grad_h = np.array(grad(self.barrier, argnums=0)(x,agent_rad))

        for i in range(2):         
            # Try pushing x slightly (repeat 2 times in different directions)   
            if np.isnan(grad_h[0]):
                x[i] += .001
                grad_h = np.array(grad(self.barrier, argnums=0)(x,agent_rad))
            else:
                break

        lg_h = grad_h.T.dot(xdot)

        m.addConstr((lg_h)>=-k*h**p, "cbf")
        m.update()

        # Save data to plot
        self.h_hist.append(h)

    def step_toward_safety(self,m):
        x_agent = self.agent.state
        u_agent = self.agent.u
        x_obs = self.vertex.state
        u_obs = self.vertex.u
        x = x_agent - x_obs
        u = u_agent - u_obs
        xdot = self.agent.dynamics.dx(x_agent,u_agent) - self.vertex.dynamics.dx(x_obs,u_obs)
        agent_rad = self.agent.shape.radius

        grad_h = np.array(grad(self.barrier, argnums=0)(x,agent_rad))
        lg_h = grad_h.T.dot(xdot)
        # m.setObjective(grad_h.T.dot(xdot), GRB.MAXIMIZE)
        m.setObjective(m.getObjective() - 100*lg_h + u.T.dot(u), GRB.MINIMIZE)
        m.update()
    
    def plot(self, ax=None, color=None):
        """Plot the CBF value over time"""
        if ax is None:
            plt.cla()
            if color is None:
                plt.plot(self.h_hist,'-')
            else:
                plt.plot(self.h_hist,'-',color=color)
            plt.title('Barrier Function Value')
            plt.pause(.001) # Need to pause for plot to update
            plt.show()
        else:
            if color is None:
                ax.plot(self.h_hist,'-')
            else:
                ax.plot(self.h_hist,'-',color=color, linewidth=5)
        