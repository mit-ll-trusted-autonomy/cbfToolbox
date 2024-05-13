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
from cbf_toolbox.geometry import Sphere
from cbf_toolbox.vertex import Agent


class QuadCopter(Agent):
    """
    Implements a QuadCopter visualization in 3 dimensions
    """
    def __init__(self, state, radius, dynamics, safety=True, plot_arrows=False, plot_path=False, color='green', k=1.0, p=1.0):
        super().__init__(state, Sphere(radius), dynamics, safety=safety, plot_arrows=plot_arrows, plot_path=plot_path, color=None, k=k, p=p)

        if self.dynamics.n < 3 or self.dynamics.m < 3:
            raise RuntimeError('Expecting 3d position vector in first three elements of state')

        self.arm_length = radius
        self.ax = None
        self.links = dict()

    def plot(self, ax):
        self.__set_axes_links(ax)
        L = self.arm_length
        points = np.array([ [-L,0,0], [L,0,0], [0,-L,0], [0,L,0], [0,0,0], [0,0,0] ]).T
        points[0,:] += self.state[0]
        points[1,:] += self.state[1]
        points[2,:] += self.state[2]
        self.links[self.ax]['l1'].set_data(points[0,0:2],points[1,0:2])
        self.links[self.ax]['l1'].set_3d_properties(points[2,0:1])
        self.links[self.ax]['l2'].set_data(points[0,2:4],points[1,2:4])
        self.links[self.ax]['l2'].set_3d_properties(points[2,2:3])
        # avoid matplotlib bug by keeping line data as a sequency (e.g., 5:6 instead of just 5)
        # more info here: https://github.com/matplotlib/matplotlib/issues/22308
        self.links[self.ax]['hub'].set_data(points[0,5:6],points[1,5:6])
        self.links[self.ax]['hub'].set_3d_properties([points[2,5]])

    def __set_axes_links(self, ax):
        if ax == self.ax:
            return

        # set current axes
        self.ax = ax
        if ax not in self.links:
            self.links[self.ax] = {
                'l1':  self.ax.plot([],[],[],color='blue',linewidth=3,antialiased=False)[0],
                'l2':  self.ax.plot([],[],[],color='red',linewidth=3,antialiased=False)[0],
                'hub': self.ax.plot([],[],[],marker='o',color='green', markersize=6,antialiased=False)[0]
            }