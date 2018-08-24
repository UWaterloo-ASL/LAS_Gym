# Integration Demo for ROM Exhibit
Example script: [`Integration_Demo_for_ROM_Exhibit_new.py`](https://github.com/UWaterloo-ASL/LAS_Gym/blob/master/Integration_Demo_for_ROM_Exhibit_new.py)

## Assumptions
Here are the assumptions we made to integrate with `master_script.py`.
1. `observation = get_observation()`: a function to get observation(an array where each entry corresponds to an IR data.) 
```python
def get_observation()
    return observation
```
2. `take_action(action)`: a function to take action(an array where each entry corresponds to a value for that actuator.)
```python
def take_action(action)
    # execute action
```

## Instructions on How to Integrate
1. Download 
