from copy import deepcopy
import yaml
import random

def create_obstacle_yaml(base='static_180.yaml',goal_file='env_obstacle.yaml', num_obs=1):
    if num_obs == 1:
        y_loc = random.random()*40 + 5
        new_yaml_data_dict = { 'centroid_x': 33, 'centroid_y': 33, 'dx': 5, 'dy': 5,
        'orientation': 0, 'velocity_x': 0.9, 'velocity_y': 0, 'acc_x': 0, 'acc_y': 0 }

    with open(base,'r') as yamlfile:
            cur_yaml = yaml.safe_load(yamlfile)
            cur_yaml['obstacles'].append(new_yaml_data_dict)
    
    with open(goal_file,'w') as yamlfile:
            yaml.safe_dump(cur_yaml, yamlfile)
    
    return goal_file

def create_obs_dict(centroid, width, orientation, vel, acc):
    # takes in a list of tuples and create a dict
    obst_dict = None

    obs_dict = { 'centroid_x': centroid[0], 'centroid_y': centroid[1], 'dx': width[0], 'dy': width[1],
                 'orientation': orientation, 
                 'velocity_x': vel[0], 'velocity_y': vel[1], 
                 'acc_x': acc[0], 'acc_y': acc[1] }

    return obs_dict

def create_robot_dict(pos, orientation, velocity):
    robot_dict = None
    robot_dict = {'centroid_x': pos[0], 'centroid_y': pos[1],
                  'orientation': orientation, 'velocity': velocity}
    return robot_dict

def create_goal_dict(pos):
    return {'x_loc': pos[0], 'y_loc': pos[1]}

def append_to_yaml(base='static_180.yaml',goal_file='env_obstacle.yaml', dict_list=None, robot_dict=None, goal_dict=None):

    with open(base,'r') as yamlfile:
            cur_yaml = yaml.safe_load(yamlfile)
            for dict in dict_list:
                cur_yaml['obstacles'].append(dict)
            cur_yaml['robot'] = robot_dict
            cur_yaml['goal'] = goal_dict
    
    with open(goal_file,'w') as yamlfile:
            yaml.safe_dump(cur_yaml, yamlfile)
    
    return goal_file


def create_manan_env1():
    obs_dict_list = []

    # OBS1
    centroid = (50, 25)
    width      = (5, 25)
    orientation = 0 # degrees
    vel = (0, 0)
    acc = (0, 0)
    obs1 = create_obs_dict(centroid, width, orientation, vel, acc) 
    obs_dict_list.append(obs1)

    # DEFINE robot
    r_pos = (10, 25)
    orientation = 0
    velocity = 5
    robot_dict = create_robot_dict(r_pos, orientation, velocity)

    # DEFINE GOAL
    g_pos = (80, 30)
    goal_dict = create_goal_dict(g_pos)

    append_to_yaml(goal_file='manan_env1.yaml', dict_list=obs_dict_list, robot_dict=robot_dict, goal_dict=goal_dict) 

def create_manan_env2():
    obs_dict_list = []

    # OBS1
    centroid = (50, 25)
    width      = (5, 10)
    orientation = 0 # degrees
    vel = (0, 3)
    acc = (0, 0)
    obs1 = create_obs_dict(centroid, width, orientation, vel, acc) 
    obs_dict_list.append(obs1)

    # DEFINE robot
    r_pos = (10, 25)
    orientation = 0
    velocity = 5
    robot_dict = create_robot_dict(r_pos, orientation, velocity)

    # DEFINE GOAL
    g_pos = (80, 30)
    goal_dict = create_goal_dict(g_pos)

    append_to_yaml(goal_file='manan_env2.yaml', dict_list=obs_dict_list, robot_dict=robot_dict, goal_dict=goal_dict) 

def create_manan_env3():
    obs_dict_list = []

    # OBS1
    centroid = (33, 40)
    width      = (1, 50)
    orientation = 0 # degrees
    vel = (0, 0)
    acc = (0, 0)
    obs1 = create_obs_dict(centroid, width, orientation, vel, acc) 
    obs_dict_list.append(deepcopy(obs1))

    centroid = (66, 10)
    width      = (1, 50)
    orientation = 0 # degrees
    vel = (0, 0)
    acc = (0, 0)
    obs1 = create_obs_dict(centroid, width, orientation, vel, acc) 
    obs_dict_list.append(obs1)

    # DEFINE robot
    r_pos = (10, 37.5)
    orientation = 0
    velocity = 5
    robot_dict = create_robot_dict(r_pos, orientation, velocity)

    # DEFINE GOAL
    g_pos = (80, 5)
    goal_dict = create_goal_dict(g_pos)

    append_to_yaml(goal_file='manan_env3.yaml', dict_list=obs_dict_list, robot_dict=robot_dict, goal_dict=goal_dict) 


create_manan_env1()

create_manan_env2()

create_manan_env3()

