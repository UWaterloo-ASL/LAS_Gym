# LAS_Gym
Living Architecture System simulated environment with OpenAi Gym APIs

## To run the simulating environment
1. Clone the whole repositry
2. `Open scene _LAS-Scenes/livingArchitecture_simple_demo_nondistributed.ttt_ in V-REP`
3. `Run  _Interaction_Among_RedLightExcitedVisitor_LAS_Env.py_`

## Organization

1. Environment class: `RedLightExcitedVisitor_LAS_Env.py`
2. LAS Agent class: `LASAgent.py`
3. Visitor Agent class: `RedLightExcitedVisitorAgent.py`

## Features
Environment class can automatically load object names and handles as long as the scene follows the naming method with `_node#` substring.

## Dependency
1. OpenAi gym package: `pip install gym`
