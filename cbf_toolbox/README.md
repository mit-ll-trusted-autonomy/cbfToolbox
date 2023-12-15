<!-- # DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
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
# Author: Andrew Schoer, andrew.schoer@ll.mit.edu -->

### Vertex and Edge ###
*Vertex* and *Edge* are abstract base classes used to form the relationship between each other. Examples of subclasses of *Vertex* are *Agent*, *Obstacle*, and *Goal*. Examples of subclasses of *Edge* are *CBF* (control barrier function) and *CLF* (control Lyapunov function). If a *CBF* is created with an *Agent* and an *Obstacle*, then the agent will avoid that obstacle. If a *CLF* is created with an *Agent* and a *Goal*, then the agent will move to that goal.

### Vertex ###
A *Vertex* object has state, shape, and dynamics. In this code, state is a numpy array of length *n* where *n* is determined by the choice of the dynamics.

#### Geometry ####
A *Vertex* object has a shape, which is defined by one of the subclasses of the abstract class *Shape* in `geometry.py`. Each shape has a differentiable function *func*, that is used in the barrier function or the Lyapunov function constraints. The subclasses all have their own constructors depending on the information needed to define the shape.
The current list of shapes include
-Point
-Sphere
-Ellipsoid
-HalfPlane

*Some of these shapes can be written more generically to work for arbitrary dimensions.

#### Dynamics ####
The dynamics class defines the motion of a *Vertex* object. Some commonly used dynamics are preprogrammed in the `dynamics.py`, but new dynamics can also be defined programmatically. However, this assumes the dynamics to be control affine and written in the form

x' = f(x) + g(x)u

Here we define a two-dimensional single integrator.
```
from cbf_toolbox.dynamics import Dynamics
n = 2 # Dimension of the state vector
m = 2 # Dimension of the input vector u
f = lambda x : 0 # Drift function
g = lambda x : np.eye(2) # Affine control function
sing_int = Dynamics(n,m,f,g)
```
This is one of the existing dynamics subclasses in the code. The code above is equivalent to
```
from cbf_toolbox.dynamics import SingleIntegrator2d
sing_int = SingleIntegrator2d()
```
Dynamics are decoupled from the state. When the `step()` function is called, the state, *x*, is passed to the dynamics from the *Vertex* object, and the new state is recorded by the *Vertex* object.

*At this time, SingleIntegrator2d() is the only subclass of Dynamics. I would like to add SingleIntegrator3d, DoubleIntegrator2d, Unicycle. Maybe dubens vehicle

#### Agents ####
An *Agent* is a subclass of *Vertex*. At each step of the simulation, an optimization function is solved to determine the best controls for each agent. The objectives and constraints for this optimization problem are determined by the relationships between the agents and the other *Vertex* objects in the scenario. An agent requires a state, radius, and dynamics.
```
import numpy as np
from cbf_toolbox.vertex import Agent
a1 = Agent(np.array([0.,0.]), 1.0, SingleIntegrator2d())
```
*At this time, agents are spherical, but this should be generalized to any shape that is in `geometry.py`

#### Goal ####
A *Goal* is a subclass of *Vertex*. It has a state and dynamics.
```
from cbf_toolbox.vertex import Goal
g1 = Goal(np.array([5.0,0.0])
```
*Currently, the goal defaults to a *Point* *shape, but could be generalized to any shape. 
*A goal has dynamics, but currently no way to define motion.

#### Obstacles ####
An *Obstacle* is a subclass of *Vertex*. An *Obstacle* has state, shape, and dynamics.
```
import numpy as np
from cbf_toolbox.vertex import Obstacle
from cbf_toolbox.geometry import Sphere
o1 = Obstacle(np.array([1.,1.]),Sphere(.5),SingleIntegrator2d())
```
*Obstacles have dynamics, but do not currently have a way to define the motion
*HalfPlane shaped objects need to be revisited

### Edge ###
An *Edge* defines the relationship between two *Vertex* objects. The construction of *Edge*s is built into the *Simulation* object. Before the simulation begins, it will loop over the list of agents and form the proper edges to define the behavior of each agent.

#### Control Barrier Function (CBF) ####
A *CBF* is a subclass of *Edge*. It forms a safety relationship between an *Agent* and a *Vertex*. The second parameter can be any type of *Vertex*, generally either another *Agent* or an *Obstacle*. The shape of the second *Vertex* parameter has a `func()` which determines the barrier function, *h(x)*.

#### Control Lyapunov Function (CLF) ####
A *CLF* is a subclass of *Edge*. It forms an attractive relationship between an *Agent* and a *Vertex*. The second parameter can be any type of *Vertex*, generally it is a *Goal*. The shape of the second *Vertex* parameter has a `func()` which determines the Lyapunov function, *V(x)*..

#### Reference Control ####
A *ReferenceControl* is a subclass of edge. This is used to define how an agent will move when there is no *Goal*

### Simulation ###
The simulation class is where everything in the scenario is tracked. This is where the agents and obstacles in the environment live, and where the stepping through time takes place. Everything that was created above can be added to the *Simulation* object.
```
from cbf_toolbox.safety import Simulation
s = Simulation()
s.add_agent(a1,g1)
s.add_obstacle(o1)

# When everything has been added to the simulation
s.simulate(numsteps=100, dt=1.0, plotting=True)
```
If `plotting=True`, then there will show the agents and obstacles in the scenario as they develop.

Once the simulation has completed, you can also plot the value of the barrier functions and the Lyapunov functions over time.
```
# Plot CLFs
for clf in s.control:
   clf.plot()

# Plot CBFs
for cbfs in s.cbf_by_agent:
   for cbf in cbfs:
      if cbf is not None:
         cbf.plot()
```