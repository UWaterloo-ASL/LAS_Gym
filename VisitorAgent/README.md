# VisitorAgent class
This folder contains algorithms to control visitors. As mentioned in [LASAgent classes](https://github.com/UWaterloo-ASL/LAS_Gym/tree/ROM_Agent_Community_LM/LASAgent), it is necessary to maintain a separated set of algorithms for controlling visitor. On one hand, it is prohibitive to use general learning algorithm to contral visitor, since it is impossible to learn from scrath without a good understand of visitors' preference. On the other hand, if both Living Architecture System and Visitors both learn from scrath, the convergence time will be extraordinary long and even will never converge at all.

## Organization
1. **Bright-light-excited Visitor** who is excited when there is a bright light with intensity >=0.95 in LAS.
   * Implemented in: `BrightLightExcitedVisitorAgent.py`
2. **Red-light-excited Visitor** who is excited when there is a red light being trun on in LAS.
   * Implemented in: `RedLightExcitedVisitorAgent.py`
