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
3. Four variables are needed:
   * `observation_space`: a gym.spaces.Box object
   * `observation_space_name`: an array where each entroy corresponds to the name of that sensor.
   * `action_space`: a gym.spaces.Box object
   * `action_space_name`: an array where each entroy corresponds to the name of that actuator.

## Instructions on How to Integrate
1. Download 
