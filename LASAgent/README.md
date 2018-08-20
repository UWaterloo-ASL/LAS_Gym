# LASAgent classes
This folder contains implemented Learning Algorithms for intelligent agent. 


## Organization of Learning Agent Classes
To ensure reusability, we provides two intermediate classes for realistic interaction in which reward signal is not provided by environment, at the same time seamlessly working with virtual environment with interfaces as in [OpenAI Gym](https://gym.openai.com/docs/). 
#### Internal Environment for Single Agent ####
* InternalEnvOfAgent.py
#### Internal Environment for Agent Community ####
* InternalEnvOfCommunity.py
#### Agent ####

* Actor-Critic LASAgent: `LASAgent_Actor_Critic.py`
* Random action LASAgent: `RandomLASAgent.py`
