from VrepRemoteApiBindings import vrep
import numpy as np
import csv
import math
import warnings


def loadCSV():
    name = []
    coordinate = []
    height = 2
    with open('output.csv', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader) # remove first row
        for row in csv_reader:
            name.append(row[0][1:]) # remove the single quotation mark at the same time
            coordinate.append([float(row[1]), float(row[3]), height]) # use Z as Y coordinate and manually set Z as height

    return name, coordinate


def loadModel(clientID, grp_name, grp_coordinate):
    num_group = len(grp_name)

    for i in range(0, num_group):

        single_grp_name = grp_name[i]
        single_grp_coord = grp_coordinate[i]

        # find center of A and B
        centerA = [0,0,0]
        centerB = [0,0,0]
        for j in range(0,len(single_grp_name)):
            if single_grp_name[j] == 'A':
                centerA = single_grp_coord[j]
            elif single_grp_name[j] == 'B':
                centerB = single_grp_coord[j]

        # loop through all groups
        for j in range(0, len(single_grp_name)):

            if single_grp_name[j] == 'A' or single_grp_name[j] == 'B':
                res, handle = vrep.simxLoadModel(clientID, 'models/LAS_Model/center.ttm', 0, vrep.simx_opmode_blocking)
                if res != vrep.simx_return_ok:
                    print("Load model failed, " + str(res))
                    print('handle = ', handle)
                vrep.simxSetObjectPosition(clientID, handle, -1, single_grp_coord[j], vrep.simx_opmode_oneshot)
                # vrep.simxGetObject
                # for index in range(0,2):
                #     res, handle = vrep.simxGetObjectChild(clientID,handle,index,vrep.simx_opmode_blocking)


            else:
                sma_number = (int(single_grp_name[j])-1)%6 + 1
                filename = 'models/LAS_Model/edge' + str(sma_number) + '.ttm'
                res, handle = vrep.simxLoadModel(clientID, filename, 0, vrep.simx_opmode_blocking)
                if res != vrep.simx_return_ok:
                    print("Load model failed, " + str(res))
                    print('handle = ', handle)
                vrep.simxSetObjectPosition(clientID, handle, -1, single_grp_coord[j], vrep.simx_opmode_oneshot)
                if int(single_grp_name[j]) <= 6:
                    vector = np.array(single_grp_coord[j][0:2]) - np.array(centerA[0:2])
                    gamma = np.arccos(vector[0]/np.linalg.norm(vector))
                    # as the range of arccos() is [0,Pi], we need to check the case where angle is negative (y < 0)
                    if vector[1] < 0:
                        gamma = -gamma
                    # print("gamma = ", gamma*180/math.pi)
                else:
                    vector = np.array(single_grp_coord[j][0:2]) - np.array(centerB[0:2])
                    gamma = np.arccos(vector[0] / np.linalg.norm(vector))
                    if vector[1] < 0:
                        gamma = -gamma
                    # print("gamma = ", gamma * 180 / math.pi)
                vrep.simxSetObjectOrientation(clientID,handle,-1, [0,0,gamma], vrep.simx_opmode_oneshot)

def getGroup(names, coordinates):
    prev_group_num = 1
    grouped_names = []
    grouped_coord = []

    name_group = []
    coord_group = []
    for i in range(0,len(names)):
        s = str.split(names[i], '-')
        group_num = int(s[0])
        id = s[1]

        if group_num != prev_group_num:
            grouped_names.append(name_group)
            grouped_coord.append(coord_group)
            name_group = []
            coord_group = []
            prev_group_num = group_num

        name_group.append(id)
        coord_group.append(coordinates[i])

    return grouped_names, grouped_coord


if __name__ == '__main__':
    """
    Load models into the scene
    """
    clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
    if clientID != -1:
        print('Connected to remote API server')
    else:
        print('Failed connecting to remote API server')

    names, coords = loadCSV()
    grp_names, grp_coords = getGroup(names,coords)
    loadModel(clientID,grp_names,grp_coords)

    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
    vrep.simxFinish(clientID)