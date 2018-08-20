# LAS_Gym
Living Architecture System simulated environment with OpenAi Gym APIs

## To run the simulating environment


## Interaction Pattern
In our design, the interaction between LAS and Environment (i.e. `Interaction_LAS_and_Env.py`) is parallel with the interaction between Visitor and Environment (i.e. `Interaction_Visitor_and_Env.py`). These two scripts can run in different process, and the stop of interaction between Visitor and Environment will not affect the interaction between LAS and Environment.

## Organization

   
### Class
   1. Environment class for LAS Agent: `LASEnv.py`
   2. LAS Agent class: 
      * `RandomLASAgent.py`: random action
      * `LASAgent_Actor_Critic.py`: extrinsically motivated behavior
   3. Environment class for Visitor Agent: `VisitorEnv.py`
   4. Visitor Agent class: 
      * `RedLightExcitedVisitorAgent.py`: red light excited visitor
### Interaction paradigm and Simulator

<img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/InitialDesignIdeas/DesignFigures/WholePacture_Distributed_IntrinsicMotivation.png" width="400" height="400" />       <img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/InitialDesignIdeas/DesignFigures/ROM_Simulation_Scene.png" width="400" height="400" /> 

## How To Use
### Demo Interaction scripts
   1. Interaction between LAS and Environment: `Interaction_LASAgentActorCritic_and_Env.py`
   2. Interaction between Visitor and Environment: `Interaction_Visitor_and_Env.py`
   3. Interaction between **Extrinscially Motivated LASAgent** and Environment: `Interaction_LASAgentActorCritic_and_Env.py`
### Demo 1: Single Agent
   `interaction_Single_Agent_and_Env.py`
### Demo 2: Multi-Agents
   `interaction_Distributed_Agent_Community_and_Env.py`
### Demo 3: Visitors
   `Interaction_Multi_BrightLightExcitedVisitor_and_Env.py`

## Features
  1. Environment class can automatically load object names and handles as long as the scene follows the naming method with `_node#` substring.
  2. The interactions among Environment, LAS and Visitor can run in parallel.

## Dependency
   1. OpenAi gym package: `pip install gym`
   2. keras
   3. Tensorflow
   4. tflearn
