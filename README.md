# LAS_Gym
Living Architecture System simulated environment with OpenAi Gym APIs

## To run the simulating environment
### Method 1: with GUI (Only for visulizing behavior)
   1. Clone the whole repositry
   2. In V-REP: File -> Open scene -> choose _LAS-Scenes/livingArchitecture_singleVisitor_nondistributed.ttt_ 
   3. Run: `_Interaction_LAS_and_Env.py_` to start interaction between LAS and Environment
   4. Run: `_Interaction_Visitor_and_Env.py_` to start interaction between Visitor and Environment.

Notes:
  * You should start with running `_Interaction_LAS_and_Env.py_` first, then start running `_Interaction_Visitor_and_Env.py_`, because Visitor intertacts with Env by a different [temporary port in chiled-script](http://www.coppeliarobotics.com/helpFiles/en/remoteApiServerSide.htm) from defualt port.
  * In this manner, you can visualize the interactions among LAS, Env and Visitor. However, this visualization will slow down interaction dramatically due to vision render. We recommend you only use this method when you examine your agents' behavior. For other cases, please use **Method 2**. 

### Method 2: without any GUI (Recommended)
   1. Clone the whole repositry
   2. Open `Terminal` or `Command Prompt`
   3. Change directory to your vrep.exe. For example: `cd C:\Program Files\V-REP3\V-REP_PRO_EDU\`
   4. start V-REP via the command line: `vrep -h -s C:\LAS_Gym\LAS-Scenes\livingArchitecture_singleVisitor_nondistributed.ttt`

Notes:
  * For details on starting V-REP via the command line, please visit [here](http://www.coppeliarobotics.com/helpFiles/en/commandLine.htm).
  * You can also add path of vrep.exe to environment variable. If there is error to load scene, please firmly follow **Method 2**.
  * Compared with **Method 1**, **Method 2** is faster. Therefore, this method is preferred when you training your learning algorithm.

## Interaction Pattern
In our design, the interaction between LAS and Environment (i.e. `_Interaction_LAS_and_Env.py_`) is parallel with the interaction between Visitor and Environment (i.e. `_Interaction_Visitor_and_Env.py_`). These two scripts can run in different process, and the stop of interaction between Visitor and Environment will not affect the interaction between LAS and Environment.

## Organization
### Interaction scripts
   1. Interaction between LAS and Environment: `Interaction_LAS_and_Env.py`
   2. Interaction between Visitor and Environment: `Interaction_Visitor_and_Env.py`
### Class
   1. Environment class for LAS Agent: `LASEnv.py`
   2. LAS Agent class: `LASAgent.py`
   3. Environment class for Visitor Agent: `VisitorEnv.py`
   3. Visitor Agent class: `RedLightExcitedVisitorAgent.py`
### Interaction paradigm
Interaction paradigm of our design.

<img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/InitialDesignIdeas/DesignFigures/WholePacture_Distributed_IntrinsicMotivation.png" width="400" height="400" />

## Features
  1. Environment class can automatically load object names and handles as long as the scene follows the naming method with `_node#` substring.
  2. The interactions among Environment, LAS and Visitor can run in parallel.

## Dependency
   1. OpenAi gym package: `pip install gym`
   2. numpy
