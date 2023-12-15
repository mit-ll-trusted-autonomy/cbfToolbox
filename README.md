<!-- Author: Andrew Schoer, andrew.schoer@ll.mit.edu -->

# CBF Toolbox #
An easy-to-use package to add safety to dynamic control systems with control barrier functions.

## Setup Python environment with Conda
First clone the environment into your desired location
```
git clone git@github.mit.edu:iitchs/cbfToolbox.git
cd cbfToolbox/
```
Create a conda environment to run the CBF Toolbox.
```
conda env create -f ./environment.yml
conda activate cbf_toolbox
pip install -e .
```
Try running one of the examples.
```
python ./examples/simple_example.py
```

## Gurobi Setup
If there is an error with Gurobi when running the example, you may need to setup your Gurobi license. Follow instructions at this 
[link to setup Gurobi](https://support.gurobi.com/hc/en-us/articles/14799677517585).

### Contact ###
Andrew Schoer
andrew.schoer@ll.mit.edu

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
 
This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.
 
© 2023 Massachusetts Institute of Technology.
 
Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
 
The software is provided to you on an As-Is basis
 
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
