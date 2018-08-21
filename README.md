# LAS_Gym
This versatile reposity provides simulation environment with [OpenAi Gym APIs](https://gym.openai.com/docs/) for Living Architecture System developed by a colaboration with [Living Architecture System Group](http://livingarchitecturesystems.com).

## Organization
At following, we overview what are included in this reposity, and detailed information is provided by each hyper-link.
1. **[LAS-Scenes](https://github.com/UWaterloo-ASL/LAS_Gym/tree/ROM_Agent_Community_LM/LAS-Scenes):**
   * **Royal Ontario Museum (ROM) Exhibit: Transforming Space**
      * V-REP Scene: `livingArchitecture_ROM_exhibit.ttt`
2. **[Environment](https://github.com/UWaterloo-ASL/LAS_Gym/tree/ROM_Agent_Community_LM/Environment):**
   * **Living Architecture Environment**
      * LASEnv.py
   * **Visitor Environment**
      * BrightLightExcitedVisitorEnv.py
      * RedLightExcitedVisitor_LAS_Env.py
3. **[LASAgent](https://github.com/UWaterloo-ASL/LAS_Gym/tree/ROM_Agent_Community_LM/LASAgent):**
   * **Intermediate Internal Environment Classes**
      1. **Internal Environment for Single Agent** 
         * Implemented in `InternalEnvOfAgent.py`
      2. **Internal Environment for Agent Community**
         * Implemented in `InternalEnvOfCommunity.py`
   * **Learning Agent Classes**
      * **Actor-Critic LASAgent**
         * Implemented in `LASAgent_Actor_Critic.py`
      * **Random action LASAgent**
         * Implememted in `RandomLASAgent.py`
4. **[Visitor Agent class](https://github.com/UWaterloo-ASL/LAS_Gym/tree/ROM_Agent_Community_LM/VisitorAgent):** 
      * **Bright-light-excited Visitor** who is excited when there is a bright light with intensity >=0.95 in LAS.
         * Implemented in: `BrightLightExcitedVisitorAgent.py`
      * **Red-light-excited Visitor** who is excited when there is a red light being trun on in LAS.
         * Implemented in: `RedLightExcitedVisitorAgent.py`
## Interaction Pattern
In our design, the interaction between LAS and Environment is parallel with the interaction between Visitor and Environment, as shown in **Figure 1**. These two interactions can run in different process, and the stop of interaction between Visitor and Environment will not affect the interaction between LAS and Environment.

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/ROM_Agent_Community_LM/InitialDesignIdeas/DesignFigures/Interaction_Pattern.png"  /> 

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **Figure 1.** Interaction Pattern
## How To Interact With Environment
1. **For Interaction With Virtual Environment**
   * **Step 1: Run the Simulating Scene in V-REP**
   * **Step 2: Run Python Script - Interaction between LAS-Agent and Environment**
      * General Framework
         1. Instantiate LAS Environment Object
         2. Instantiate LAS Agent Object
         3. Interaction in while loop
         * Example Script: 
            * Non-distributed Single Giant LAS Agent: `Interaction_Single_Agent_and_Env.py` 
            * Distributed Multi-agent: `Interaction_Distributed_Agent_Community_and_Env.py`
            * Random LAS Agent: `Interaction_RandomLASAgent_and_Env.py`
   * **Step 3: Run Python Script - Interaction between Visitor-Agent and Environment**
      * General Framework
         1. Instantiate Visitor Environment Object
         2. Instantiate Visitor Agent Object
         3. Interaction in while loop
         * Example Scritp:
            * Bright-light-excited Visitor Agent: `Interaction_Multi_BrightLightExcitedVisitor_and_Env.py`
### For Interaction With Real Environment
For LAS Agent, the only difference when interacting with real environment is in the receiving of **observation** and delivering of **action**. And for real environment, visitor is physical humanbody. Therefore, we only need to consider **Python Script - Interaction between LAS-Agent and Environment**.
* General Framework:
   1. Instantiate LAS Agent Object
   2. Interaction in while loop
* Overall framework for this script:
```python
        # Instatiate LAS-Agent
        agent = InternalEnvOfAgent(...)
        # Interaction loop
        while not done:
            if x_order_MDP == 1:
                # Generate action
                actionLAS = agent.interact(observation, reward, done)
                # Take action in realy system, and retrive new observation
                $$$$<Note: integration "take_action(actionLAS)">$$$$
                $$$$<Note: integration "observation = get_observation()">$$$$
            elif x_order_MDP > 1:
                # Feed (x_order_MDP-1) observation
                for obs_temp_i in range(x_order_MDP-1):
                    # the first obs is the immediate obaservation afer taking action
                    if obs_temp_i == 0: 
                        agent.feed_observation(observation)
                    else:
                        $$$$<Note: integration "observation = get_observation()">$$$$
                        agent.feed_observation(observation)
                # The last obs is input into interact function.
                observation = get_observation()
                actionLAS = agent.interact(observation, reward, done)
                # Take action in realy system, and retrive new observation
                $$$$<Note: integration "take_action(actionLAS)">$$$$
                $$$$<Note: integration "observation = get_observation()">$$$$
            else:
                raise Exception('Please choose a proper x_order_MDP!')
```

## Features
  1. Environment class can automatically load object names and handles as long as the scene follows the naming method with `_node#` substring.
  2. The interactions among Environment, LAS and Visitor can run in parallel.

## Dependency
   1. OpenAi gym package: `pip install gym`
   2. keras
   3. Tensorflow
   4. tflearn
