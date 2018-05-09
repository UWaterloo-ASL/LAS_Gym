# LAS_Gym
Living Architecture System simulated environment with OpenAi Gym APIs

## To run the simulating environment
### Method 1: 
1. Clone the whole repositry
2. In V-REP: File -> Open scene -> choose _LAS-Scenes/livingArchitecture_simple_demo_nondistributed.ttt_ 
3. Run: `_Interaction_LAS_ans_Env.py_` to start interaction between LAS and Environment
4. Run: `_Interaction_Visitor_and_Env.py_` to start interaction between Visitor and Environment.

Note:
  * You should start running `_Interaction_LAS_ans_Env.py_` first, then start running `_Interaction_Visitor_and_Env.py_`, because Visitor intertacts with Env by a different temporary port from defualt port.
  
  * In this manner, you can visualize the interactions among LAS, Env and Visitor. However, this visualization will slow down interaction dramatically due to vision render. We recommend you only use this method when you examine your agents' behavior. For other cases, please use *Method 2*. 

## Organization

1. Environment class: `RedLightExcitedVisitor_LAS_Env.py`
2. LAS Agent class: `LASAgent.py`
3. Visitor Agent class: `RedLightExcitedVisitorAgent.py`

## Features
Environment class can automatically load object names and handles as long as the scene follows the naming method with `_node#` substring.

## Dependency
1. OpenAi gym package: `pip install gym`
