# Integration Demo for ROM Exhibit
Example script: [`Integration_Demo_for_ROM_Exhibit_new.py`](https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/Integration_Demo_for_ROM_Exhibit_new.py)

## Assumptions
Here are the assumptions we made to integrate with `master_script.py`.
1. `observation = get_observation()`: a function to get observation(an array where each entry corresponds to an IR data and the range of observation value is **[0, 1]**.) 
```python
def get_observation()
    return observation
```
2. `take_action(action)`: a function to take action(an array where each entry corresponds to a value for that actuator and the range of action value is **[-1, 1]**.)
```python
def take_action(action)
    # execute action
```
3. Four variables are needed: (We can handle this, but need you to tell us which sensor or actuator corresponds to which entroy in an obervation array and action array.)
   * `observation_space`: a gym.spaces.Box object
   * `observation_space_name`: an array where each entroy corresponds to the name of that sensor.
   * `action_space`: a gym.spaces.Box object
   * `action_space_name`: an array where each entroy corresponds to the name of that actuator.
4. In our simulation, we also assume the physical exhibit is like this:
   [V-REP Scene For Royal Ontario Museum (ROM) Exhibit: Transforming Space](https://github.com/UWaterloo-ASL/LAS_Gym/tree/master/LAS-Scenes#v-rep-scene-for-royal-ontario-museum-rom-exhibit-transforming-space)
   
&nbsp; &nbsp; <img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/InitialDesignIdeas/ROM_Exhibit/Single_Giant_Agent.png" width="400"  />     &nbsp;  <img src="https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/InitialDesignIdeas/ROM_Exhibit/Agent_Community_Partition.png" width="400"  /> 

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **Figure 1.** The Name of Each Node in ROM Exhibit

## Instructions on How to Integrate
1. Download `LASAgent` folder

