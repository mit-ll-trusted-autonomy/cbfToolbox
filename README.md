<!-- Author: Andrew Schoer, andrew.schoer@ll.mit.edu -->

# Control Barrier Function Toolbox #
A Python package that makes control barrier functions simple.

## Setup Python environment with Conda
First clone the environment into your desired location
```
git clone git@github.com:mit-ll-trusted-autonomy/cbfToolbox.git
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

## Acknowledgements
Roberto Tron, Boston University - Advisor to CBF Toolbox development\
Guang Yang\
Max Cohen - [cbfToolbox.jl](https://github.com/maxhcohen/CBFToolbox.jl)\
Ahmad Ahmad\
Amy Fang\
Helena Teixera-Dasilva\
Cristian So

## Distribution and Disclaimer Statements

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

© 2023 Massachusetts Institute of Technology.

    Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
    SPDX-License-Identifier: BSD-3-Clause

This material is based upon work supported by the Under Secretary of Defense for 
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any 
opinions, findings, conclusions or recommendations expressed in this material 
are those of the author(s) and do not necessarily reflect the views of the Under 
Secretary of Defense for Research and Engineering.

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. 
Government rights in this work are defined by DFARS 252.227-7013 or 
DFARS 252.227-7014 as detailed above. Use of this work other than as specifically 
authorized by the U.S. Government may violate any copyrights that exist in this work.

The software/firmware is provided to you on an As-Is basis
