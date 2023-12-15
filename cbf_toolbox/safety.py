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

from cbf_toolbox.edge import ReferenceControl, CLF, CBF
import matplotlib
import mpl_toolkits.mplot3d.axes3d as Axes3D

from cbf_toolbox.vertex import Goal
matplotlib.use("TkAgg")
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
from gurobipy import GRB, Model, LinExpr

class Simulation(object):
    """A Simulation object keeps track of all agents and obstacles being simulated"""

    def __init__(self, stop_time=None):
        self.stop_time = stop_time
        
        self.agents = list()
        self.obsts = list()
        self.control = list() # List of length len(self.agents) with either the control vector or the CLF object for each agent
        self.cbf_by_agent = list() # list of lists for all CBFs for each agent
        
        self.m = Model('clf_cbf_qp')
        self.m.Params.LogToConsole = 0 #Stop optimizer from publishing results to console

        self.step_num = 0
        
        self.ready_to_sim = False

    def add_agent(self, agent, control, upper_bounds=1.0, lower_bounds=-1.0):
        """Adds an agent to the simulation. Must provide either a goal or u_ref
    
        If control is type Goal, then a CLF controller is created
        If control is a list or np.array, then it is treated as u_ref
        """
        self.agents.append(agent)
        agent.add_grb_control_vars(self.m, upper_bounds=upper_bounds, lower_bounds=lower_bounds)
        
        if type(control) is Goal:
            self.control.append(CLF(agent,control))
            agent.goal = control
            agent.u_ref = None
        else:
            self.control.append(ReferenceControl(agent,np.array(control)))
            agent.goal = None
            agent.u_ref = control

    def add_obstacle(self,obst):
        """Adds an obstacle to the simulation"""

        if self.ready_to_sim:
            raise RuntimeError('Can''t add obstacles once simulation has started')
        self.obsts.append(obst)

    def build_cbfs(self):
        """Creates the CBFs objects for the simulation for agent-agent pairs and agent-obstacle pairs"""
        for agent in self.agents:
            a_cbfs = []
            if agent.safety:
                for obst in self.agents:
                    if agent is not obst:
                        a_cbfs.append(CBF(agent,obst))

                for obst in self.obsts:
                    a_cbfs.append(CBF(agent,obst))
            else:
                a_cbfs.append(None) # Needed to set agent objective with safety off

            self.cbf_by_agent.append(a_cbfs)

    def build_grb_model(self):
        """Builds the Gurobi model at each time step"""
        # Reset gurobi model
        self.m.reset()

        # Remove the constraints and objective
        self.m.setObjective(LinExpr())
        self.m.remove(self.m.getConstrs())
        self.m.remove(self.m.getQConstrs())
        self.m.update()

        for control in self.control:
            # Add CLF
            if isinstance(control,CLF):
                control.add_clf(self.m)

        # Add CBFs for each agent and its obstacles
        for idx,cbfs in enumerate(self.cbf_by_agent):
            for cbf in cbfs:
                if cbf is None and self.agents[idx].goal is None:
                    # If safety is off and no goal is set, this forces agent to take given control action
                    [self.m.addConstr(self.agents[idx].control[i] == u) for i,u in enumerate(self.control[idx][self.step_num])]
                    self.m.update()
                elif cbf is not None:
                    cbf.add_cbf(self.m)

            # Add CBF objective for agent if needed
            if not isinstance(self.control[idx],CLF):
                agent = self.agents[idx]
                u = agent.u
                u_ref = self.control[idx].u_ref
                cost_func = (u-u_ref).T.dot(u-u_ref)
                self.m.setObjective(self.m.getObjective() + cost_func, GRB.MINIMIZE)
                

    def solve(self):
        """Solves the Gurobi model"""
        self.m.optimize()
        self.m.update()
        if self.m.SolCount == 0:
            print('No solution?')

    def step(self, dt=0.1):
        """Solves the Gurobi model and moves time forward one step"""
        if not self.ready_to_sim:
            self.build_cbfs()
            self.ready_to_sim = True

        self.build_grb_model()
        self.solve()
        self.step_num += 1

        if self.m.SolCount == 0:
            print('No solution')

        solution = self.m.getVars()

        for a in self.agents:
            a.step(dt=dt)

            if a.goal is not None:
                a.goal.step(dt=dt)

        for o in self.obsts:
            o.step(dt=dt)

        return solution

    def simulate(self, num_steps=100, dt=0.1, video_name=None, plotting=True):
        """Starts the simulation. Records sim to .mp4 if video_name is provided"""
        # self.plot()
        
        make_video = video_name is not None
        if make_video:
            path = pathlib.Path().resolve()
            video_path = path / 'videos'
            video_path.mkdir(exist_ok=True)
            print(f'Saving video to {video_path}')

            fig1 = plt.gcf()
            metadata1 = dict(title='CBFToolbox', artist='Matplotlib',comment='Movie support!')
            writer1 = FFMpegWriter(fps=25, metadata=metadata1)
            writer1.setup(fig1, str(video_path / f'{video_name}.mp4'), 1000)
        
        for i in range(num_steps):
            self.step()
            
            if plotting:
                self.plot()

            if make_video:
                writer1.grab_frame()

        safe_actions = []
        for a in self.agents:
            safe_actions.append(a.u)

        return safe_actions

    def plot(self):
        """Plots a single instance of time of the simulation"""
        plt.cla()
        plt.gca().axis('equal')

        [a.plot() for a in self.agents]
        [o.plot() for o in self.obsts]

        # if hasattr(self,'xlim'):
        #     plt.xlim(self.xlim)
        #     plt.ylim(self.ylim)

        plt.title("Steps: {}".format(self.step_num))

        plt.pause(.01)

    def set_plot_lims(self,xlim,ylim):
        """Sets the axis plot limits"""
        self.xlim = xlim
        self.ylim = ylim

class Simulation3d(Simulation):
    def __init__(self,
                 stop_time=None,
                 title='3d Sim',
                 xlim=(-2., 2.),
                 ylim=(-2., 2.),
                 zlim=(0, 5.)):
        super().__init__(stop_time=stop_time)
        self.title = title
        self.fig = plt.figure()
        self.ax = Axes3D.Axes3D(self.fig)
        self.ax.set_xlim3d(xlim)
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d(ylim)
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d(zlim)
        self.ax.set_zlabel('Z')
        self.fig.suptitle(f'{self.title}\nSteps: {self.step_num}')

    def plot(self):
        [a.plot(self.ax) for a in self.agents]
        [o.plot(self.ax) for o in self.obsts]

        self.fig.suptitle(f'{self.title}\nSteps: {self.step_num}')

        plt.pause(.01)