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
from cbf_toolbox.geometry import Sphere, Ellipsoid
from cbf_toolbox.dynamics import Dynamics, SingleIntegrator2d
from cbf_toolbox.vertex import Agent, Obstacle, Goal
from cbf_toolbox.safety import Simulation

def main():
    
    # Dynamics determine how the Vertex objects move
    # Some Dynamics are predefined, like SingelIntegrator2d()
    single_int = SingleIntegrator2d()

    # Users can also define dynamics programatically.
    # Let's define a model that drifts along the positive y-axis
    n = 2 # Size of the state vector
    m = 2 # Size of the input vector
    f = lambda x : np.array([0,0.5]) # Drift function
    g = lambda x : np.zeros([2,2]) # Control function
    drifting_up = Dynamics(n,m,f,g)

    # Here we define the shapes that we will use
    sphere_half = Sphere(radius=0.5)
    ellipse = Ellipsoid([1.0,0.5],45)

    # Now with dynamics defined and shapes defined, we create the Vertex objects
    state1 = np.array([0.,0.])
    state2 = np.array([5.,4.])
    a1 = Agent(state=state1, radius=0.5, dynamics=single_int)
    a2 = Agent(state=state2, radius=0.5, dynamics=single_int)
    o1 = Obstacle(state=[2.,0.], shape=ellipse, dynamics=drifting_up)
    g1 = Goal(state2)
    g2 = Goal(state1)

    # Now we can add everything to a simulation object
    s = Simulation()
    s.add_agent(agent=a1, control=g1)
    s.add_agent(agent=a2, control=g2)
    s.add_obstacle(obst=o1)

    # When everything is added, we can call the simulate function
    # Before running the simulation, the function will loop over all the agents and obstacles and create
    # the proper Edges to connect the Vertex objects (the CBF and CLF objects)
    s.simulate(num_steps=100, dt = 0.1)

    # When the simulation is over, we can examine the barrier function and Lyapunov function values over time
    fig, axs = plt.subplots(2,1, sharex=True)
    for clf in s.control:
        clf.plot(ax=axs[0])
    axs[0].set_title('Control Lyapunov Function Values')

    for cbfs in s.cbf_by_agent:
        for cbf in cbfs:
            cbf.plot(ax=axs[1])
    axs[1].set_title('Control Barrier Function Values')
    plt.show()

if __name__ == '__main__':
    main()