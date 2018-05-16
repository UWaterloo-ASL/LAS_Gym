# This folder contains scripts for automatically loading and assembling models to create a simultion scene for LAS.

1. In V-REP design a model, and save it as a model following the instructions given [here](http://www.coppeliarobotics.com/helpFiles/en/models.htm).
2. Load model by scripts `vrep_load_object_setObjectPosition.py` to specific locations given by `model_locations.csv`. 
  * Python [API](http://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsPython.htm) for load model `vrep.simxLoadModel` and set object position `vrep.simxSetObjectPosition`
