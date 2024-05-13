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
import matplotlib.pyplot as plt
from cbf_toolbox.geometry import Sphere, Ellipsoid, HalfPlane
from cbf_toolbox.dynamics import Dynamics, SingleIntegrator2d
from cbf_toolbox.vertex import Agent, Obstacle, Goal
from cbf_toolbox.safety import Simulation

def main():
    
    # Now with dynamics defined and shapes defined, we create the Vertex objects
    a1 = Agent(state=np.array([0.,0.]), shape=Sphere(0.5), dynamics=SingleIntegrator2d())
    o1 = Obstacle(state=np.array([2.,0.]), shape=HalfPlane(np.array([-1,0]), rotation=30), dynamics=SingleIntegrator2d())
    g1 = Goal(np.array([5.,0.]))
    
    # Now we can add everything to a simulation object
    s = Simulation()
    s.add_agent(agent=a1, control=g1)
    s.add_obstacle(obst=o1)

    # When everything is added, we can call the simulate function
    # Before running the simulation, the function will loop over all the agents and obstacles and create
    # the proper Edges to connect the Vertex objects (the CBF and CLF objects)
    s.simulate(num_steps=100, dt = 0.1)

    s.plot_functions()

if __name__ == '__main__':
    main()