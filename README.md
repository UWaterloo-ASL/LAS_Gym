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
2. **For Interaction With Real Environment**: For LAS Agent, the only difference when interacting with real environment is in the receiving of **observation** and delivering of **action**. And for real environment, visitor is physical humanbody. Therefore, we only need to consider **Python Script - Interaction between LAS-Agent and Environment**.
   * General Framework:
      1. Instantiate LAS Agent Object
      2. Interaction in while loop
   * Overall framework for this script:
```python
        # Instatiate LAS-Agent
        agent = InternalEnvOfAgent(...)
        try:
            # Interaction loop
            while True:
                $$$$<Note(Integration):  "observation = get_observation()">$$$$
                take_action_flag, action = agent.feed_observation(observation)
                if take_action_flag == True:
                    $$$$<Note(Integration): "take_action(action)">$$$$
        except KeyboardInterrupt:
            agent.stop()
```

## Meta-Data Produced by LAS Learning Algorithm
When interacting with real or virtual environment, all data will be saved in directory `../ROM_Experiment_results/` i.e. sub-directory `ROM_Experiment_results` of the parent directory of `interaction_script`.
* **Organization Meta-Data**
   1. For non-distributed single giant agent:
      * ROM_Experiment_results
         * LAS_Single_Agent
            * interaction_data
            * models
            * summary
   2. For distributed multi-agent:
      * ROM_Experiment_results
         * LAS_Agent_Community
            * interaction_data
         * LAS_Agent_Community_agent_1
            * interaction_data
            * models
            * summary
         * LAS_Agent_Community_agent_2
            * interaction_data
            * models
            * summary
         * LAS_Agent_Community_agent_3
            * interaction_data
            * models
            * summary
* **Visualize Meta-Data**: The visualization of meta-data is done by utilizing [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard).
   * General: `tensorboard --logdir name1:/path/to/logs/1,name2:/path/to/logs/2` (For more details on how to use tensorboard, please check `tensorboard --helpfull`)
   * Example: `tensorboard --logdir agent_community_agent1:path/to/ROM_Experiment_results/LAS_Agent_Community_agent_1/summary,agent_community_agent2:path/to/ROM_Experiment_results/LAS_Agent_Community_agent_2/summary,agent_community_agent3:path/to/ROM_Experiment_results/LAS_Agent_Community_agent_3/summary,agent_community_agent3:path/to/ROM_Experiment_results/LAS_Agent_Community_agent_1/summary,agent_community_agent2:path/to/ROM_Experiment_results/LAS_Agent_Community_agent_3/summary,single_agent:path/to/ROM_Experiment_results/LAS_Single_Agent/summary`

## Dependency
   1. [OpenAi gym](https://gym.openai.com/docs/#installation) package: `pip install gym`
   2. [Tensorflow](https://www.tensorflow.org/install/)
   3. [Keras](https://keras.io/#installation): `sudo pip install keras`
   4. [tflearn](http://tflearn.org/installation/)
