# LASAgent classes
This folder contains implemented Learning Algorithms for Living Architecture System. In theory, **Living Architecture Sytem** and **Visitor** could share the same set of Learning Algorithms to realize control of intelligent agent. However, in practice, using learning algorithm to control visitor is very hard and complex. Therefore, in our implementation, we separatively maintain control algorithms for [Living Architecture System](https://github.com/UWaterloo-ASL/LAS_Gym/tree/ROM_Agent_Community_LM/LASAgent) and [Visitor](https://github.com/UWaterloo-ASL/LAS_Gym/tree/ROM_Agent_Community_LM/VisitorAgent).

## Intermediate Internal Environment Classes
To ensure reusability, we provides two intermediate classes for realistic interaction in which reward signal is not provided by environment, at the same time seamlessly working with virtual environment with interfaces as in [OpenAI Gym](https://gym.openai.com/docs/). 
1. **Internal Environment for Single Agent**
   * InternalEnvOfAgent.py
2. **Internal Environment for Agent Community**
   * InternalEnvOfCommunity.py

## Learning Agent Classes

1. **Actor-Critic LASAgent**
   * Implemented in `LASAgent_Actor_Critic.py`
2. **Random action LASAgent**
   * Implememted in `RandomLASAgent.py`
