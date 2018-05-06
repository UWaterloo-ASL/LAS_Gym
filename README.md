# LAS_Gym
Living Architecture System simulated environment with OpenAi Gym APIs

## To run the simulating environment
1. `Open scene _livingArchitecture_simple_demo_nondistributed.ttt_ in V-REP`
2. `Run  _LAS_Env_new.py_`

## LAS_Env_new.py
Dependency: 
1. `vrep.py`
2. `vrepConst.py`
3. `remoteApi.dylib`(Mac) or `remoteApi.dll`(Windows) or `remoteApi.so`(Linux) (depend on your OS) 

(Put these three files in the same folder with _LAS_Env_new.py_)

## Features
_LAS_Env_new.py_ is recommended, because _LAS_Env_new.py_ can automatically load object names and handles as long as the scene follows the naming method with `_node#` substring.

