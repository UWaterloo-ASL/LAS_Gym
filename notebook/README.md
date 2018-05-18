# Note for Living Architecture System Projects

## Test Actor-Critic on simulation

### Large State and Action Space
**Scene:**`LAS-Scenes/livingArchitecture_singleVisitor_nondistributed_large_size.ttt`

This is run on scene with large state and action space. Specifially,

**Observation**: 
*  proximity sensor: 39 * 1, range: 0 or 1
* light color: 39 * 3, range: (0,1)

**Action**: 
* sma: 39 * 1, range: (0,1)
* light color: 39 * 3, range: (0,1)

**Learning rate**
* actor lr : 0.0001
* critic lr: 0.0001

**Exploration method**
* random action noise
* epsilon-greedy
* ~~Boltzmann Approach~~(only for discrete action)
* Bayesian Approach
* Intrinsic Motivation

**Scene Image and [Results](https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/notebook/notebook_LASAgent_Actor_Critic.ipynb)** 

<img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/notebook/images/large_LAS.png" alt="Scene Image" width="400" height="250"> <img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/notebook/images/large_LAS_results_2569episodes.png" alt="Results" width="400" height="250">


### Small State and Action Space
**Scene:** `LAS-Scenes/livingArchitecture_singleVisitor_nondistributed_small_size.ttt`

**Observation**: 
*  proximity sensor: 15 * 1, range: 0 or 1
* light color: 15 * 3, range: (0,1)

**Action**: 
* sma: 15 * 1, range: (0,1)
* light color: 15 * 3, range: (0,1)

**Learning rate**
* actor lr : 0.0001
* critic lr: 0.0001

**[Exploration method](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf)**
* random action noise
* epsilon-greedy
* ~~Boltzmann Approach~~(only for discrete action)
* Bayesian Approach
* Intrinsic Motivation

**Scene Image and [Results](https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/notebook/notebook_LASAgent_Actor_Critic_SamllSize_System.ipynb)**

<img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/notebook/images/small_LAS.png" alt="Scene Image" width="400" height="250"> <img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/notebook/images/small_LAS_results_3000episodes.png" alt="Results" width="400" height="250"> 
[<img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/notebook/images/small_LAS_results_3000episodes_video_image.png" alt="Results" width="400" height="250">](https://youtu.be/NEdSqGTIL5U)

