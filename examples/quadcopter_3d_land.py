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

from cbf_toolbox.agent_models.quadcopter import QuadCopter
from cbf_toolbox.dynamics import SingleIntegrator3d
from cbf_toolbox.safety import Simulation3d
from cbf_toolbox.vertex import Goal


if __name__ == '__main__':
    single_int = SingleIntegrator3d()
    init_state = np.array([2., 2., 4.5]) # x, y, z coordinates
    a1 = QuadCopter(state=init_state, radius=0.3, dynamics=single_int)
    g = Goal(np.array([0., 0., 0.]), dynamics=SingleIntegrator3d())
    s = Simulation3d()
    s.add_agent(agent=a1, control=g)

    s.simulate(num_steps=100, dt = 0.1)