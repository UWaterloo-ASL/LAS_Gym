# Living Architecture System Scenes
This folder contains all simulation scenes and python scripts to create a scene automatically.

### V-REP Scene For Royal Ontario Museum (ROM) Exhibit: Transforming Space
1. V-REP Scene: `livingArchitecture_ROM_exhibit.ttt`
2. Components of ROM Exhibit Scene:
   * Nodes: 24 in total
   * Each node:
      * Actuator: 1 Light + 6 SMA
      * Sensor: 1 IR Sensor
3. Overall:
   * Observation Space: 1×24 dimensions (continuous value)
   * Action Space : 7×24=168 dimensions (continuous value)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/ROM_Agent_Community_LM/InitialDesignIdeas/ROM_Exhibit/ROM_exhibit.jpg" width="300"  />     &nbsp;  <img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/ROM_Agent_Community_LM/InitialDesignIdeas/ROM_Exhibit/ROM_exhibit_simulator.png" width="300"  /> 

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **Figure 1.** ROM Exhibit: Transforming Space

### Create Scene Automatically by Loading and Assembling Models (Optional)
V-REP provide APIs to automatically loading and assembing models to create a scene. You are not necessary to know this part, only if you want to create your own scene.
#### General Steps to Create a Scene ####
1. Load object shape designed by AutoCAD etc. to make simulated object more realistic. For detailed supported file types, please refer to [Importing and exporting shapes](http://www.coppeliarobotics.com/helpFiles/en/importExport.htm) provided by V-REP.
2. In V-REP, first design a model, then save it as a model following the instructions on [Models in V-REP](http://www.coppeliarobotics.com/helpFiles/en/models.htm).
3. Load model by scripts `vrep_load_object_setObjectPosition.py` with specific locations given by `model_locations.csv`. 
   * Python [API](http://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsPython.htm) for load model `vrep.simxLoadModel` and set object position `vrep.simxSetObjectPosition`
#### Our Script to Create ROM Exhibit Scene
Here is our script to create ROM Exhibit scene `livingArchitecture_ROM_exhibit.ttt`.
* models: `las_scene_models`
* Python Script to load and assemble models: `load_sculpture.py`
* Position information of each model when assembling: `output.csv`
With these files, to create a scene of living architecture environment:
   1. Put the LAS_Gym\LAS-Scenes\las_scene_models\LAS_Model folder in your VREP model folder.  For example: C:\Program Files\V-REP3\V-REP_PRO_EDU\models\LAS_Model
   2. Open VREP and load LAS_Gym\LAS-Scenes\livingArchitecture_load_model_test.ttt  This file has one copy of all LAS Models inside. This is to make sure that the models added later has correct suffix (#num). 
   3. Run load_sculpture.py. This will read coordinates of actuators from output.csv file and set it in the scene.
   4. Delete the models without suffix （#num) and save the scene as new scene
