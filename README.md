# LAS_Gym
This versatile reposity provides simulation environment with [OpenAi Gym APIs](https://gym.openai.com/docs/) for Living Architecture System developed by a colaboration with [Living Architecture System Group](http://livingarchitecturesystems.com).

### Organization
1. **[LAS-Scenes](https://github.com/UWaterloo-ASL/LAS_Gym/tree/ROM_Agent_Community_LM/LAS-Scenes):**
   * **Royal Ontario Museum (ROM) Exhibit: Transforming Space**
      1. V-REP Scene: `livingArchitecture_ROM_exhibit.ttt`
      2. Components of ROM Exhibit Scene:
         * Nodes: 24 in total
         * Each node:
            * Actuator: 1 Light + 6 SMA
            * Sensor: 1 IR Sensor
      3. Overall:
         * Observation Space: 1×24 dimensions (continuous value)
         * Action Space : 7×24=168 dimensions (continuous value)
2. **[Environment](https://github.com/UWaterloo-ASL/LAS_Gym/tree/ROM_Agent_Community_LM/Environment):**
   * **Living Architecture Environment**
      * LASEnv.py
   * **Visitor Environment**
      * BrightLightExcitedVisitorEnv.py
      * RedLightExcitedVisitor_LAS_Env.py
3. **[LASAgent](https://github.com/UWaterloo-ASL/LAS_Gym/tree/ROM_Agent_Community_LM/LASAgent):**
   * **Actor-Critic LASAgent**
      * Implemented in `LASAgent_Actor_Critic.py`
   * **Random action LASAgent**
      * Implememted in `RandomLASAgent.py`
4. **[Visitor Agent class](https://github.com/UWaterloo-ASL/LAS_Gym/tree/ROM_Agent_Community_LM/VisitorAgent):** 
      * **Bright-light-excited Visitor** who is excited when there is a bright light with intensity >=0.95 in LAS.
         * Implemented in: `BrightLightExcitedVisitorAgent.py`
      * **Red-light-excited Visitor** who is excited when there is a red light being trun on in LAS.
         * Implemented in: `RedLightExcitedVisitorAgent.py`
### Interaction Pattern
In our design, the interaction between LAS and Environment is parallel with the interaction between Visitor and Environment. These two interactions can run in different process, and the stop of interaction between Visitor and Environment will not affect the interaction between LAS and Environment.
<img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/InitialDesignIdeas/DesignFigures/WholePacture_Distributed_IntrinsicMotivation.png" width="400" height="400" />       <img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/InitialDesignIdeas/DesignFigures/ROM_Simulation_Scene.png" width="400" height="400" /> 

## How To Use
### For Interaction With Virtual Environment

### For Interaction With Real Environment

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
