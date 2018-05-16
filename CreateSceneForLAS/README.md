# This folder contains scripts for automatically loading and assembling models to create a simultion scene for LAS.

  1. Load obect shape designed by AutoCAD etc. to make simulated object more realistic. For detailed supported file types, please refer to [instructions given here](http://www.coppeliarobotics.com/helpFiles/en/importExport.htm).
  2. In V-REP design a model, and save it as a model following the instructions given [here](http://www.coppeliarobotics.com/helpFiles/en/models.htm).
  3. Load model by scripts `vrep_load_object_setObjectPosition.py` to specific locations given by `model_locations.csv`. 
    * Python [API](http://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsPython.htm) for load model `vrep.simxLoadModel` and set object position `vrep.simxSetObjectPosition`
